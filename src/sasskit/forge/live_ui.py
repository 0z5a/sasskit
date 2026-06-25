"""Live Rich terminal UI for the forge optimization loop."""
from __future__ import annotations
import time
from typing import Optional
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import sys
# Force terminal output even when running in background/pipe
# so progress is visible when monitoring the output file
console = Console(force_terminal=True, width=100, highlight=False)


def _ns_bar(ns: float, baseline: float, width: int = 20) -> str:
    """ASCII bar showing ns/op relative to baseline."""
    if baseline <= 0:
        return ""
    ratio = ns / baseline
    filled = int(min(ratio, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{'green' if ratio < 1 else 'red'}]{bar}[/] {ratio*100:.0f}%"


class ForgeUI:
    """Live Rich dashboard for forge progress."""

    def __init__(self, name: str, baseline_ns: float, target_ns: float = 0):
        self.name = name
        self.baseline_ns = baseline_ns
        self.target_ns = target_ns or baseline_ns
        self.start_time = time.time()
        self.results: list[dict] = []
        self.best: Optional[dict] = None
        self._live: Optional[Live] = None
        self._console = Console(force_terminal=True, width=100, highlight=False)

    def start(self):
        self._live = Live(self._render(), console=self._console,
                          refresh_per_second=4, screen=False)
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None
        self._print_final_summary()

    def update(self, result: dict):
        self.results.append(result)
        if result.get("status") == "PASS" and result.get("ns_per_op"):
            if self.best is None or result["ns_per_op"] < self.best["ns_per_op"]:
                self.best = result
        if self._live:
            self._live.update(self._render())

    def _render(self):
        layout = Layout()
        layout.split_column(
            Layout(self._make_header(),    name="header",  size=3),
            Layout(self._make_stats(),     name="stats",   size=8),
            Layout(self._make_history(),   name="history", size=15),
        )
        return layout

    def _make_header(self):
        elapsed = time.time() - self.start_time
        h, m, s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
        n = len(self.results)
        title = f"[bold cyan]FORGE: {self.name}[/]  [{h:02d}:{m:02d}:{s:02d}]  [{n} iterations]"
        return Panel(Text.from_markup(title), style="blue")

    def _make_stats(self):
        table = Table.grid(padding=(0,2))
        table.add_column(justify="right", style="bold")
        table.add_column()

        best_ns = self.best["ns_per_op"] if self.best else None
        passes = sum(1 for r in self.results if r.get("status") == "PASS")
        wrongs = sum(1 for r in self.results if r.get("status") == "WRONG")
        crashes = sum(1 for r in self.results if r.get("status") in ("CRASH","TIMEOUT","ASM_ERROR"))

        table.add_row("Baseline:", f"[yellow]{self.baseline_ns:.1f} ns/op[/]")
        if best_ns:
            speedup = self.baseline_ns / best_ns
            color = "green" if speedup > 1.02 else "yellow"
            table.add_row("Best:", f"[{color}]{best_ns:.2f} ns/op  {speedup:.3f}×  "
                         f"({self.best.get('n_instructions',0)} insns, R{self.best.get('max_register',0)})[/]")
            table.add_row("vs baseline:", _ns_bar(best_ns, self.baseline_ns))
        else:
            table.add_row("Best:", "[dim]no PASS yet[/]")
        table.add_row("Pass / Wrong / Err:",
                      f"[green]{passes}[/] / [yellow]{wrongs}[/] / [red]{crashes}[/]")
        return Panel(table, title="[bold]Performance[/]", border_style="green")

    def _make_history(self):
        table = Table(show_header=True, header_style="bold cyan",
                      border_style="dim", expand=True)
        table.add_column("#", width=5)
        table.add_column("Status", width=8)
        table.add_column("ns/op", width=10)
        table.add_column("vs base", width=8)
        table.add_column("insns", width=6)
        table.add_column("regs", width=5)
        table.add_column("detail", overflow="fold")

        recent = self.results[-12:]
        for r in reversed(recent):
            it = str(r.get("iteration", "?"))
            st = r.get("status", "?")
            ns = r.get("ns_per_op")
            ni = r.get("n_instructions", 0)
            mr = r.get("max_register", 0)
            err = r.get("error_detail", "")[:50]

            if st == "PASS":
                status_str = "[green]PASS[/]"
                ns_str = f"[green]{ns:.2f}[/]" if ns else "[dim]?[/]"
                is_best = (r is self.best)
                ns_str += " ⭐" if is_best else ""
                vs = f"{ns/self.baseline_ns*100:.0f}%" if ns else "-"
                vs_color = "green" if ns and ns < self.baseline_ns else "yellow"
                vs_str = f"[{vs_color}]{vs}[/]"
            elif st == "WRONG":
                status_str = "[yellow]WRONG[/]"
                ns_str = vs_str = "[dim]-[/]"
                err = "carry/reduction bug"
            else:
                status_str = f"[red]{st}[/]"
                ns_str = vs_str = "[dim]-[/]"

            table.add_row(it, status_str, ns_str, vs_str,
                          str(ni), f"R{mr}", f"[dim]{err}[/]")

        return Panel(table, title="[bold]Recent iterations[/]", border_style="blue")

    def _print_final_summary(self):
        self._console.print()
        self._console.print(Panel(
            f"[bold]FORGE COMPLETE: {self.name}[/]\n"
            f"Baseline: [yellow]{self.baseline_ns:.1f} ns/op[/]\n" +
            (f"Best:     [green]{self.best['ns_per_op']:.2f} ns/op  "
             f"{self.baseline_ns/self.best['ns_per_op']:.3f}×  "
             f"({self.best['n_instructions']} insns)[/]"
             if self.best else "[red]No PASS found[/]") +
            f"\nIterations: {len(self.results)}  "
            f"Time: {time.time()-self.start_time:.0f}s",
            style="bold green" if self.best else "red"
        ))

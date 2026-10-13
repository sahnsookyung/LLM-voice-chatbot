/** Fixed simulation steps, adaptive drawing, and bounded work after a stall. */
export class MotionClock {
  readonly step = 1 / 60;
  private simulation = 0;
  private presentation = 0;
  reset() { this.simulation = this.presentation = 0; }
  advance(seconds: number, moving: boolean) {
    const elapsed = Number.isFinite(seconds) ? Math.max(0, Math.min(seconds, .1)) : 0;
    this.simulation += elapsed;
    this.presentation += elapsed;
    const steps = Math.floor((this.simulation + 1e-9) / this.step);
    this.simulation = Math.max(0, this.simulation - steps * this.step);
    const interval = moving ? this.step : 1 / 30;
    // WebKit timestamps can alternate 16/17 ms at 60 Hz. A strict boundary
    // skips the early callback, then discards two intervals on the late one.
    // Allow 1 ms of presentation jitter and retain the debt to avoid speeding up.
    const tolerance = .001;
    const draw = this.presentation + tolerance >= interval;
    if (draw) this.presentation -= Math.floor((this.presentation + tolerance) / interval) * interval;
    return {steps, draw};
  }
}

"""PsychoPy n-back task integration with EEG recording."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..core.procedure import make_procedure
from ..core.recorder import EEGRecorder
from ..core.settings import (
    BG_COLOR,
    ITI,
    LETTER_HEIGHT,
    LETTERS,
    MATCH_PROB,
    RESPONSE_KEY,
    SHOW_FEEDBACK_DEFAULT,
    STIM_DURATION,
    TRIAL_DURATION,
)


def _make_sequence(n: int, n_trials: int) -> list[tuple[str, bool]]:
    seq: list[tuple[str, bool]] = []
    for t in range(n_trials):
        if t >= n and random.random() < MATCH_PROB:
            seq.append((seq[t - n][0], True))
        else:
            options = [
                letter for letter in LETTERS
                if t < n or letter != (seq[t - n][0] if t >= n else None)
            ]
            # Fallback safety.
            if not options:
                options = list(LETTERS)
            seq.append((random.choice(options), False))
    return seq


def _quit_experiment(win, fh=None):
    if fh is not None:
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass
    try:
        win.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        from psychopy import core

        core.quit()
    except Exception:  # noqa: BLE001
        pass


def run_session(
    subject_id: str,
    session_idx: int,
    driver,
    montage,
    out_dir: str | Path = "data",
    *,
    rng: random.Random | None = None,
    fullscreen: bool = True,
    show_feedback: bool = SHOW_FEEDBACK_DEFAULT,
    min_duration: int = 30,
    max_duration: int = 90,
    n_blocks: int = 3,
    window_size: tuple[int, int] = (1920, 1080),
) -> Path:
    """Run one recorded n-back session and write its output folder."""
    # Lazy import so the package stays importable without a display.
    from psychopy import core, event, visual

    from ..io import write_session

    if rng is None:
        rng = random.Random()

    blocks = make_procedure(
        rng=rng,
        n_blocks=n_blocks,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    win = visual.Window(
        fullscr=fullscreen,
        size=window_size,
        color=BG_COLOR,
        units="height",
        allowGUI=False,
    )

    letter_stim = visual.TextStim(
        win, text="", height=LETTER_HEIGHT, color="white", font="Arial"
    )
    fixation = visual.TextStim(win, text="+", height=0.1, color="white")
    message = visual.TextStim(
        win, text="", height=0.04, color="white", wrapWidth=1.4, alignText="center"
    )
    frame = visual.Rect(
        win, width=0.3, height=0.3, lineWidth=6, fillColor=None
    )

    recorder = EEGRecorder(driver, montage)

    def show_message(text, key_list=("space",)):
        message.text = text
        message.draw()
        win.flip()
        event.clearEvents()
        keys = event.waitKeys(keyList=list(key_list) + ["escape"])
        if keys and "escape" in keys:
            raise KeyboardInterrupt

    try:
        recorder.start()

        # Welcome (cut out).
        recorder.set_state(None)
        show_message(
            "Welcome!\n\n"
            "The experiment consists of several blocks.\n"
            "The block types, order and duration are randomized.\n\n"
            "Press SPACE to begin."
        )

        for block in blocks:
            block_name = block.name
            duration = block.duration
            occurrence = block.occurrence
            n = block.n

            if block_name == "rest":
                # Instruction (cut out).
                recorder.set_state(None)
                show_message(
                    "REST\n\n"
                    "Please look at the cross in the centre of the screen and relax.\n"
                    "Try to keep your head still.\n\n"
                    "Press SPACE to start."
                )
                event.clearEvents()
                fixation.draw()

                rest_clock = core.Clock()
                # First flip marks block onset (recorded state).
                win.callOnFlip(recorder.set_state, block_name)
                win.callOnFlip(rest_clock.reset)
                win.flip()

                while rest_clock.getTime() < duration:
                    fixation.draw()
                    win.flip()
                    keys = event.getKeys(["escape"])
                    if "escape" in keys:
                        raise KeyboardInterrupt

                # End of block (cut out before feedback).
                recorder.set_state(None)

            else:
                # Instruction (cut out).
                recorder.set_state(None)
                show_message(
                    f"{n}-BACK\n\n"
                    f"A series of letters will appear one at a time.\n"
                    f"Press '{RESPONSE_KEY.upper()}' when the current letter is the SAME as the letter "
                    f"shown {n} {'letter' if n == 1 else 'letters'} earlier.\n"
                    f"Otherwise, do not press anything.\n\n"
                    f"Respond as quickly and accurately as you can.\n\n"
                    f"Press SPACE to start."
                )

                n_trials = int(duration / TRIAL_DURATION)
                seq = _make_sequence(n, n_trials)

                trial_clock = core.Clock()

                for i, (letter, is_match) in enumerate(seq, start=1):
                    letter_stim.text = letter
                    event.clearEvents()

                    responded = False
                    rt = None
                    feedback_until = None
                    feedback_color = None

                    letter_stim.draw()

                    if i == 1:
                        win.callOnFlip(recorder.set_state, block_name)
                    win.callOnFlip(trial_clock.reset)
                    win.flip()

                    while True:
                        t = trial_clock.getTime()
                        if t >= TRIAL_DURATION:
                            break
                        if t < STIM_DURATION:
                            letter_stim.draw()
                        if (
                            feedback_until is not None
                            and t < feedback_until
                        ):
                            frame.lineColor = feedback_color
                            frame.draw()
                        win.flip()

                        keys = event.getKeys(
                            [RESPONSE_KEY, "escape"], timeStamped=trial_clock
                        )
                        for key, key_rt in keys:
                            if key == "escape":
                                raise KeyboardInterrupt
                            if key == RESPONSE_KEY and not responded:
                                responded = True
                                rt = key_rt
                                if show_feedback:
                                    feedback_color = "green" if is_match else "red"
                                    feedback_until = rt + 0.2

                    # (Trial result not persisted in session output; EEG is the target.)

                recorder.set_state(None)

                if show_feedback:
                    # Compute simple counts for feedback (re-derive quickly).
                    pass
                show_message(
                    f"End of the {n}-back block\n\n"
                    f"Trials in total: {n_trials}\n\n"
                    f"Press SPACE to continue."
                )

        recorder.set_state(None)
        show_message(
            "The experiment is finished.\n\n"
            "Thank you!\n\n"
            "Press SPACE to exit."
        )

    except (KeyboardInterrupt, Exception) as exc:
        recorder.set_state(None)
        try:
            recorder.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            win.close()
        except Exception:  # noqa: BLE001
            pass
        raise

    # Stop and collect.
    recorder.stop()
    signal, time_arr, state_changes = recorder.get_recording()

    meta = {
        "session_id": f"{subject_id}_{session_idx}",
        "subject_id": subject_id,
        "electrodes": list(montage.names),
        "sfreq": float(montage.sfreq),
        "n_channels": int(montage.n_channels),
    }

    session_dir = write_session(
        out_dir=out_dir,
        session_idx=session_idx,
        meta=meta,
        signal=signal,
        time=time_arr,
        state_changes=state_changes,
    )

    try:
        win.close()
    except Exception:  # noqa: BLE001
        pass

    return session_dir

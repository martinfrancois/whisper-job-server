# Contributing

Bug reports, fixes and docs improvements are welcome, especially from people who run Queue Whisperer on an operating system, GPU or Python version I haven't tried. Leave private recordings, transcripts and tokens out of issues and pull requests.

## Three ways to contribute

I want your time to go into pull requests that don't collide with someone else's work and that I can accept. So every contribution takes one of these paths:

- **Small fix**: a bug fix, a typo, a docs improvement or a similar small change needs no issue. Open a pull request straight away. If an issue for it already exists, claim that issue first as described in [Solve an issue](#solve-an-issue).
- **Existing issue**: claim it with `/take` and wait for the assignment before you open a pull request (see [Solve an issue](#solve-an-issue)).
- **New feature**: start a thread in the [Ideas discussions](https://github.com/martinfrancois/whisper-job-server/discussions/categories/ideas). Once I approve the feature there, I open an issue for it, and you claim that issue with `/take`. Please don't open a pull request for a feature without an approved issue that is assigned to you.

## Solve an issue

Browse the [existing issues](https://github.com/martinfrancois/whisper-job-server/issues) to find one that interests you. I add the `help wanted` label to issues where community contributions are especially welcome and the `good first issue` label for tasks suited to new contributors.

Before you start on an issue, check that nobody is assigned to it and that no open pull request already addresses it. The issue's **Development** section in the sidebar lists the linked pull requests. Then claim it by commenting `/take` on a line of its own, and wait until you're assigned before you open a pull request. If the issue has the `help wanted` or `good first issue` label, nobody holds it and no open pull request addresses it, a bot assigns it to you right away; otherwise it replies with the reason. Nobody can take an issue that someone else holds, and you can hold up to 3 issues at a time (set in the [claim workflow](.github/workflows/claim-issue.yml)). For issues without those labels, leave a comment asking if you can work on it and I'll get back to you. If you ask in plain words on a labelled issue ("can I take this?"), the bot answers once and tells you whether `/take` would work.

If you haven't opened a pull request for the issue 14 days after you were assigned (or after your last comment there), the bot asks in the issue whether you're still working on it. Reply there or open a pull request, a draft is fine; if neither happens within 7 days of that reminder, the bot unassigns you so someone else can take the issue. Once you have an open or merged pull request for the issue, there are no reminders. If your pull request is closed without merging, the 14 days start again from that day. The [claim reminders workflow](.github/workflows/claim-reminders.yml) sets both waiting times. If you stop working on an issue, say so in a comment so someone else can take it.

A pull request that closes an issue its author doesn't hold, or one that another open pull request already addresses, gets a comment and the `claim conflict` label.

## Run the checks

CI runs these on Python 3.11, 3.12 and 3.13 with Node.js 24. The tests replace the ML packages with fakes, so they need no GPU and no model download. The worker tests need `ffmpeg` and `ffprobe` on the PATH and are skipped without them.

```bash
pip install -r requirements-dev.txt
python -m pytest -rs
node --test tests/test_ui.cjs
python examples/make_wav.py /tmp/tone.wav
cmp /tmp/tone.wav examples/tone.wav
```

**Coffee on me.** If you've had a pull request merged here and we meet in person, at a conference or anywhere else, tell me your GitHub handle and I'll buy you a coffee or a drink. It's one per person, however many pull requests you've made. It's a thank-you, not payment: the project doesn't pay for contributions or offer bounties.

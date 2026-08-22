# Running the benchmark on a remote x86 machine

SWE-bench ships official Docker images built for `linux/amd64` only. On an Apple
Silicon Mac they run under emulation, which costs roughly 3–5× wall time and makes
a 30-instance run an overnight job instead of an evening one. Any native x86_64
box with Docker is a large win — including an ordinary gaming laptop.

The GPU is irrelevant here: inference happens in the Claude API, so the local
machine only needs cores, RAM and disk.

## Sizing

| Resource | Needed | Note |
|---|---|---|
| CPU | 4+ cores | `--workers` should be about half your core count; test suites are mostly single-threaded but the grading phase is not. |
| RAM | 16 GB | Comfortable at `--workers 4`. Each container is light during the agent phase and heavier while grading. |
| Disk | **the real constraint** | Instance images are 2–6 GB each and share base layers per repository. A 30-instance subset spanning ~10 repos lands around 60–120 GB. Prune between runs. |

Keeping a subset to fewer distinct repositories dramatically cuts the image
footprint, which is worth doing for the first pilot run.

## Windows host: WSL2 + Docker Engine

Docker Engine inside WSL2 is preferable to Docker Desktop here — lighter, no
licence question, and the daemon is already Linux-native.

**1. Install WSL2** (PowerShell as Administrator):

```powershell
wsl --install -d Ubuntu-24.04
```

**2. Cap WSL's memory** so Windows keeps headroom. Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=12
```

**3. Enable systemd** inside Ubuntu so the Docker service starts on boot.
Edit `/etc/wsl.conf`:

```ini
[boot]
systemd=true
```

Then `wsl --shutdown` from PowerShell and reopen Ubuntu.

**4. Install Docker Engine** inside Ubuntu:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

Log out and back in, then confirm with `docker run --rm hello-world`.

**5. Stop the laptop sleeping** mid-run (PowerShell as Administrator):

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

## Reaching it from the Mac

Tailscale is the least painful option: it works across networks, needs no port
forwarding or firewall rules, and survives the laptop changing IP.

Inside Ubuntu:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh
```

Install Tailscale on the Mac too, sign into the same account, then:

```bash
ssh <user>@<machine-name>
```

If you would rather not add a dependency, the alternative is `openssh-server`
inside WSL plus a `netsh interface portproxy` rule on Windows — but the WSL IP
changes on every boot, so the proxy rule has to be refreshed each time.

## Running

```bash
git clone https://github.com/Darrenus/codeloop.git && cd codeloop
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[eval]"
export ANTHROPIC_API_KEY=...

# smoke test first: 5 instances, no grading, confirm patches come out non-empty
python eval/run_swebench.py --n 5 --workers 2 --run-name smoke --skip-grading 2>/dev/null || \
python eval/run_swebench.py --n 5 --workers 2 --run-name smoke

# then the real thing, detached so an SSH drop does not kill it
tmux new -s bench
python eval/ablate.py --n 30 --workers 4 --tag v1
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t bench`. The runner is
resumable — rerunning with `--resume` skips instances already in
`predictions.json`.

## Reclaiming disk

Between arms, and certainly between runs:

```bash
docker container prune -f
docker image prune -a -f
docker system df          # check what is left
```

WSL2's virtual disk does not shrink on its own. To reclaim the space on Windows
after pruning:

```powershell
wsl --shutdown
Optimize-VHD -Path "$env:LOCALAPPDATA\Packages\<distro>\LocalState\ext4.vhdx" -Mode Full
```

## Pulling results back

```bash
rsync -avz <user>@<machine>:codeloop/eval/results/ ./eval/results/
```

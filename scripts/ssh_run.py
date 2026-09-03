import os
import paramiko, sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    os.environ["SSH_HOST"],
    port=int(os.environ.get("SSH_PORT", "22")),
    username=os.environ["SSH_USER"],
    password=os.environ.get("SSH_PASSWORD"),
    timeout=20,
)
cmd = " ".join(sys.argv[1:]) or "echo CONNECTED; hostname; whoami; uname -a"
stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
sys.stdout.write(out)
if err:
    sys.stderr.write(err)
client.close()

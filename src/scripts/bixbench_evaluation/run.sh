echo "Starting dataset server"
run_expt server.yaml &

echo "Waiting for servers to start..."
while ! curl -s localhost:8042 >/dev/null 2>&1; do
    sleep 5
    echo "Waiting for first server on port 8042..."
done
echo "First server is running"
echo "Starting runners"

run_expt runner.yaml &

# Single-VM cloud deployment

This deployment preserves the API/worker shared SQLite queue and artifact volume on one host. It adds HTTPS with Caddy and an nginx admission limit (one submission/minute/IP, burst two). No API, database, or worker port is directly exposed.

## Server choice

Start with Ubuntu 24.04 LTS on x86-64, 4 vCPUs, 8 GB RAM, and at least 40 GB disk. These are starting resources to measure, not a load-tested capacity guarantee. Keep one chemistry worker. Native packages and the Docker build still need verification on the selected Linux host.

DigitalOcean, Hetzner, AWS EC2/Lightsail, and a lab-owned Linux server can all run this Compose stack. A server already administered by the lab avoids opening a new paid account. GitHub Pages remains suitable for the static website, not the chemistry jobs. Providers with separate ephemeral API/worker filesystems need a different queue/storage design before deployment.

## Install and launch

On a newly provisioned VM with Docker Engine and the Docker Compose plugin installed using the provider's or Docker's official instructions:

```bash
git clone https://github.com/BaratiLab/smiles-forcefield.git
cd smiles-forcefield
cp deploy/cloud.env.example deploy/cloud.env
# Set API_DOMAIN to a hostname whose DNS A record points to this VM.
# Choose whether this is anonymous access or a shared-key lab beta.
nano deploy/cloud.env
docker compose --env-file deploy/cloud.env -f deploy/compose.cloud.yaml up --build -d
```

Allow inbound TCP 80/443 and SSH from the administrator's address. UDP 443 is optional HTTP/3. Do not expose port 8000. Review the actual provider invoice before creating the server: VM fees can be supplemented by tax, backups, IPv4, storage, or transfer fees.

## Verify before sharing

```bash
curl https://YOUR_API_DOMAIN/api/health
curl -X POST https://YOUR_API_DOMAIN/api/jobs \
  -H 'Content-Type: application/json' -d '{"smiles":"CCCC"}'
# Poll the returned job URL until succeeded, then download all.zip.
docker compose --env-file deploy/cloud.env -f deploy/compose.cloud.yaml logs --tail=100 worker
```

If an API key is configured, send X-API-Key on job/status/download requests. Run all five development molecules again on Linux, and verify archives and native energy/force reports. Native NAMD is still not included.

Use the Pages website's Connection settings to set `https://YOUR_API_DOMAIN`, or distribute the website link with `?api=https://YOUR_API_DOMAIN`. The API also serves the same website directly at that HTTPS hostname, so a separate Pages frontend is optional. The CORS origin is configured for `https://baratilab.github.io`.

## Operations

Back up the `jobs` Docker volume. Its files include molecular inputs/results, so limit administrator access. The worker deletes terminal artifact directories after 24 hours by default. Queue records remain in SQLite. Monitor disk usage, job failures, and queue length; avoid scaling to multiple VMs until the queue/storage design changes. A forcibly terminated worker can leave a running record, as described in the root README.

Pause access before upgrading dependencies. Run Docker builds on the server, check both services, and repeat the five-molecule smoke test before reopening submissions. This scaffold does not automatically purchase a server, create DNS records, or manage billing.

# Explanation behind IoTB and LLM Implementations

## IoTB Implementation

The Apache IoTDB layer is implemented as a dedicated time-series persistence backend for TSGuartd. It operates on multivariate environmental time series indexed by timestamp and sensor, supports hierarchial sensor path model and aligned measurement groups. The CSV remains the audit/experiment artifact layer, IoTDB becomes the canonical queryable time-series store for simulation outputs.

It is deployed as a separate standalone Docker service, not embedded into Streamlit and not mixed into the Python process.

### Architecture and Infrastructure

The implementation includes the following Docker and data management components:

- **docker-compose.iotdb.yml** - Docker configuration for IoTDB containerization
- **docker/iotdb/data/** - Persistent volume mount for time-series data storage
- **docker/iotdb/logs/** - Container logs directory for monitoring and debugging

The integration is config file / environment only, and the added configuration files are:

- **.env.example**
- **.env**
- **integrations/iotdb/config.py**

Important IoTDB variables:

- **TSGUARD_IOTDB_ENABLED=true** - Whether IoTDB writing is active
- **TSGUARD_IOTDB_HOST=127.0.0.1** 
- **TSGUARD_IOTDB_PORT=6667** 
- **TSGUARD_IOTDB_USER=root**
- **TSGUARD_IOTDB_PASSWORD=root**
- **TSGUARD_IOTDB_DATABASE=root.tsguard** - Which logical database is used
- **TSGUARD_IOTDB_DATASET=pm25** - Which dataset is used
- **TSGUARD_IOTDB_MODEL_VERSION=model_TSGuard.pth** - Which model version label is attached to records
- **TSGUARD_IOTDB_IGNORE_LATE=true** - Whether repeated late writes should be ignored

A dedicated integration package is added:

- **integrations/iotdb/__init__.py**
- **integrations/iotdb/client.py** - loads validated settings from .env
- **integrations/iotdb/config.py** - opens the native IoTDB session
- **integrations/iotdb/schema.py** - defines path-building and schema creation
- **integrations/iotdb/writer.py** - handles canonical writes
- **integrations/iotdb/run_views.py** - reconstructs stored runs into analysis-friendly DataFrames

Each sensor is modeled as an IoTDB device path, and each stored timestamp contains an aligned group of measurements describing the accepted value and its provenance.

The final path design is:

**root.tsguard.<dataset>.<run_id>.sensor_<sensor_id>**

For each sensor and timestamp, the aligned schema stores:

1. value: final numerical value that TSGuard accepted for that sensor at that timestamp
2. source_kind: provenance class of the accepted value
3. constraint_flag: whether the final accepted value is associated with a constraint issue
4. strategy: how the value is produced
5. model_version: model identifier attached to the write

### Canonical Write Policy

IoTDB stores only the final accepted canonical output stream.

1. original real values are stored
2. accepted imputed values are stored
3. fallback-revised values are stored
4. rejected intermediate candidates are not stored
5. audit history remains outside IoTDB

If the same dataset is simulated multiple times, the original timestamps are identical. Without run namespacing, different experiments would collide in the same device paths. So each simulation run gets a generated run ID using:

dataset + timestamp

Format:

**<dataset>_run_YYYY_MM_DD_HH_MM_SS**

A separate readback layer is added to support inspection and future streamlit extensions.

**integrations/iotdb/run_views.py**
**scripts/show_run_values.py**

Role of **run_views.py** - This module is the reusable analysis layer. It provides functions to list stored runs, summarize runs, load one run as a long DataFrame, rebuild wide views, export CSVs.

Role of **show_run_values.py** - This is the CLI inspection script. It lists available run IDs, lets the user choose one, lets the user choose one of two views, prints the selected reconstruction, exports it to outputs/run_views/.

This is designed so that a future Streamlit page can import the same backend functions rather than duplicating logic. And they support two output views:

1. View A - Imputed / Fallback only
2. View B - Final canonical dataset

## LLM Implementation

Replaced the old chatbot’s purely rule-based reply generator with a hybrid conversational layer:
1. primary backend: remote LLM via Groq
2. fallback backend: existing rule-based assistant

The UI remains the same, with the only modifications done to the reply-generation path.

### Architecture and Infrastructure

A separate backend package was added:

- **integrations/llm/__init__.py**
- **integrations/llm/config.py**
- **integrations/llm/context.py** 
- **integrations/llm/providers.py**
- **integrations/llm/service.py**

Important LLM Variables:

- **GROQ_API_KEY=your_real_groq_api_key_here** 
- **TSGUARD_LLM_ENABLED=true** - whether LLM mode is enabled
- **TSGUARD_LLM_PROVIDER=groq** - which provider is active
- **TSGUARD_LLM_MODEL=llama-3.1-8b-instant** - which model is used
- **TSGUARD_LLM_TIMEOUT_SECONDS=20** - timeout behavior
- **TSGUARD_LLM_TEMPERATURE=0.2** - generation behavior
- **TSGUARD_LLM_MAX_COMPLETION_TOKENS=350**
- **TSGUARD_LLM_MAX_HISTORY_MESSAGES=8** - how much chat history is retained
- **TSGUARD_LLM_ENABLE_FALLBACK=true** - whether fallback is allowed

*Groq* is used as the remote API provider. 

For context, the assistant receives a compact structured summary of the live TSGuard State to maintain low latency, reduction to privacy exposure and lower hallucination risk.

The backend relies on a structured snapshot object - **SystemSnapshot**. It contains fields such as:

1. whether simulation is running
2. whether training is active
3. current page
4. processed timestamp count
5. total timestamp count
6. current simulated timestamp
7. missing count at the current step
8. imputed count at the current step
9. sample affected sensors
10. global missing percentage
11. constraint sensitivity
12. recent neighbor alerts
13. recent constraint alerts
14. simulation speed
15. sigma threshold
16. missing-value thresholds
17. baseline sensor count
18. dynamic captor count

When asked about a particular sensor, the LLM layer adds a dedicated sensor-detail block, containing things like:

1. sensor identity resolution
2. current value if available
3. whether it is real or imputed
4. percent originally missing so far
5. percent imputed so far
6. summary statistics over processed values

## Installation

#### Downloadables and Python Environment

1. Clone the repository

```bash
git clone https://github.com/kaaaziz/ts_guard.git
cd ts_guard
```

2. Install and verify Docker

For macOS, install Docker Desktop and start it. Or you can start it from:

```bash
open -a Docker
```

Wait until Docker is fully running, then verify:

```bash
docker --version
docker compose version
docker run hello-world
```

For Linux (Ubuntu-based distributions):

```bash
#Install Docker

sudo apt-get update
sudo apt-get install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable and Start Docker:
sudo systemctl enable --now docker

# Verify installation:
docker --version
docker compose version
sudo docker run hello-world
```

3. Create and activate a virtual environment

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

4. Install Python dependencies

```bash
pip install -r requirements.txt
```

5. Install PyTorch separately only if needed

CPU-only:

```bash
pip install torch torchvision torchaudio
```

CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```pytorch.org/whl/cu118
```

CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

6. Verify the Python environment

```bash
python -c "import torch; import streamlit; import pandas; print('Installation successful')"
```

7. Create the environment configuration file

If `.env.example` exists:

```bash
cp .env.example .env
```

Then edit `.env` and set at least the IoTDB and LLM values you need.

#### IoTDB startup and verification

8. Check the Docker Compose configuration for IoTDB

```bash
docker compose -f docker-compose.iotdb.yml config
```

9. Start IoTDB

```bash
docker compose -f docker-compose.iotdb.yml up -d
```

10. Check that the IoTDB container is running

```bash
docker ps --filter name=tsguard-iotdb
```

11. Initialize IoTDB from the TSGuard scripts

```bash
python scripts/init_iotdb.py
```

12. Verify IoTDB schema / connectivity

```bash
python scripts/verify_iotdb.py
```

---

#### Run the TSGuard application

13. Start the Streamlit app

```bash
streamlit run main_app.py
```

If you prefer running it through Python explicitly:

```bash
python -m streamlit run main_app.py
```

---

#### Inspect stored experiment runs

14. Open the stored-run viewer

```bash
python scripts/show_run_values.py
```

15. What the script does

* lists all available run IDs stored in IoTDB
* lets you choose one run
* lets you choose one of two views:

  * A = imputed / fallback only
  * B = full final canonical values
* prints the chosen view in wide format
* saves the result to `outputs/run_views/`

16. Output files produced by the run-view script

Examples:

```bash
outputs/run_views/<run_id>__imputed_only.csv
outputs/run_views/<run_id>__canonical_all.csv
```

---

#### Minimal command sequence


1. Activate venv

```bash
source venv/bin/activate
```

2. Start IoTDB

```bash
docker compose -f docker-compose.iotdb.yml up -d
```

3. Verify IoTDB quickly

```bash
python scripts/init_iotdb.py
python scripts/verify_iotdb.py
```

4. Start TSGuard

```bash
streamlit run main_app.py
```

5. Inspect saved runs later

```bash
python scripts/show_run_values.py
```

---

#### Stop commands

17. Stop only the Streamlit app

Press:

```bash
Ctrl + C
```

18. Stop IoTDB but keep persisted data

```bash
docker compose -f docker-compose.iotdb.yml down
```

19. Stop IoTDB and remove Docker volumes for a completely clean database reset

```bash
docker compose -f docker-compose.iotdb.yml down -v --remove-orphans
```

20. Start again after stopping

```bash
docker compose -f docker-compose.iotdb.yml up -d
python scripts/init_iotdb.py
python scripts/verify_iotdb.py
```

---

#### Clean reproducible reset for a fresh experiment

Use this only when completely clean IoTDB state rerun is needed.

21. Full clean reset

```bash
docker compose -f docker-compose.iotdb.yml down -v --remove-orphans
docker rm -f tsguard-iotdb 2>/dev/null || true
docker compose -f docker-compose.iotdb.yml up -d
python scripts/init_iotdb.py
python scripts/verify_iotdb.py
```

After this, previously stored IoTDB runs are removed from the database.

---

### Minimal troubleshooting commands

A. Check the Python environment

1. Confirm the active Python executable

```bash
which python
python -V
```

2. Confirm required modules import correctly

```bash
python -c "import streamlit, pandas, torch; print('Python environment OK')"
```

---

B. Check Docker / IoTDB status

3. Check Docker Compose config

```bash
docker compose -f docker-compose.iotdb.yml config
```

4. Check the IoTDB container

```bash
docker ps -a --filter name=tsguard-iotdb
```

5. Check IoTDB container logs

```bash
docker compose -f docker-compose.iotdb.yml logs -f iotdb
```

6. Restart IoTDB cleanly without deleting volumes

```bash
docker compose -f docker-compose.iotdb.yml down
docker compose -f docker-compose.iotdb.yml up -d
```

---

C. If IoTDB does not initialize

7. Run init and verify again

```bash
python scripts/init_iotdb.py
python scripts/verify_iotdb.py
```

8. If the container looks stale, remove it and recreate it

```bash
docker rm -f tsguard-iotdb 2>/dev/null || true
docker compose -f docker-compose.iotdb.yml up -d
```

---

D. If you want to inspect IoTDB manually from inside the container

9. Open a shell in the IoTDB container

```bash
docker exec -it tsguard-iotdb bash
```

10. Start the IoTDB CLI

```bash
/iotdb/sbin/start-cli.sh -h 127.0.0.1 -p 6667 -u root -pw root
```

11. Useful SQL commands inside the CLI

```sql
SHOW DATABASES;
SHOW TIMESERIES root.tsguard.**;
SELECT ** FROM root.tsguard.** LIMIT 20;
```

---

E. If the run viewer shows no runs

12. First verify that runs exist in IoTDB

```bash
python scripts/verify_iotdb.py
```

13. Then run the viewer again

```bash
python scripts/show_run_values.py
```

If no runs appear, it usually means one of these:

* simulation has not yet written data
* IoTDB is disabled in `.env`
* the database is reset with `down -v`

---

F. If Streamlit runs but IoTDB writing does not happen

14. Check that IoTDB is enabled in `.env`

```bash
grep TSGUARD_IOTDB_ENABLED .env
```

15. Check that the dataset/database settings are present

```bash
grep TSGUARD_IOTDB_ .env
```

16. Re-run init and verify

```bash
python scripts/init_iotdb.py
python scripts/verify_iotdb.py
```

---

G. If the LLM assistant does not answer through Groq

17. Check that the Groq API key is loaded

```bash
grep GROQ_API_KEY .env
```

18. Check LLM-related settings

```bash
grep TSGUARD_LLM_ .env
```

19. Restart Streamlit after changing `.env`

```bash
streamlit run main_app.py
```

import simpy
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# LOAD DATA


patients = pd.read_csv(
    r"E:\New Volume(D)\OUSL 3\EEX 5362 Performance Modelling\MP\622512695_Final_Simulation\patient.csv"
)

service_log = pd.read_csv(
    r"E:\New Volume(D)\OUSL 3\EEX 5362 Performance Modelling\MP\622512695_Final_Simulation\service.csv"
)


patients["arrival_time"] = pd.to_datetime(patients["arrival_time"])
service_log["service_start_time"] = pd.to_datetime(service_log["service_start_time"])
service_log["service_end_time"] = pd.to_datetime(service_log["service_end_time"])


# CONSULTATION TIME LOOKUP

consult = service_log[service_log["service_type"] == "consultation"].copy()
consult["consult_time"] = (
    consult["service_end_time"] - consult["service_start_time"]
).dt.total_seconds() / 60

consult_time_map = dict(zip(consult["patient_id"], consult["consult_time"]))


# ARRIVAL PREPARATION

patients = patients.sort_values("arrival_time")
start_time = patients["arrival_time"].min()

patients["Arrival_Min"] = (
    patients["arrival_time"] - start_time
).dt.total_seconds() / 60


# SAFE MEAN

def safe_mean(values):
    return statistics.mean(values) if values else 0


# SIMULATION FUNCTION

def run_simulation(patients_df, consult_time_map, num_doctors, label):

    waiting_times = []
    consultation_times = []
    total_times = []

    queue_time = []
    queue_length = []

    doctor_busy_time = 0

    env = simpy.Environment()
    doctors = simpy.Resource(env, capacity=num_doctors)

    # Queue monitor
    def monitor_queue(env, resource):
        while True:
            queue_time.append(env.now)
            queue_length.append(len(resource.queue))
            yield env.timeout(1)

    def patient(env, consult_time):
        nonlocal doctor_busy_time

        arrival = env.now
        with doctors.request() as req:
            yield req
            start = env.now

            wait = start - arrival
            waiting_times.append(wait)
            consultation_times.append(consult_time)

            doctor_busy_time += consult_time
            yield env.timeout(consult_time)

            total_times.append(env.now - arrival)

    def hospital(env, data):
        for _, row in data.iterrows():
            yield env.timeout(row["Arrival_Min"] - env.now)
            consult_time = consult_time_map.get(row["patient_id"], 15)
            env.process(patient(env, consult_time))

    SIM_TIME = int(patients_df["Arrival_Min"].max() + 60)

    env.process(hospital(env, patients_df))
    env.process(monitor_queue(env, doctors))
    env.run(until=SIM_TIME)

    utilization = doctor_busy_time / (num_doctors * SIM_TIME)

    return {
        "label": label,
        "doctors": num_doctors,
        "patients_arrived": len(patients_df),
        "patients_served": len(total_times),
        "avg_wait": safe_mean(waiting_times),
        "min_wait": min(waiting_times) if waiting_times else 0,
        "max_wait": max(waiting_times) if waiting_times else 0,
        "avg_total": safe_mean(total_times),
        "utilization": utilization,
        "queue_time": queue_time,
        "queue_length": queue_length
    }


# DATASETS

patients_normal = patients.copy()
patients_peak = pd.concat([patients, patients], ignore_index=True)
patients_peak["Arrival_Min"] = patients_peak["Arrival_Min"].sort_values().values


# DOCTORS

BASE_DOCTORS = int(input("Enter number of doctors (normal load): "))
IMPROVED_DOCTORS = BASE_DOCTORS * 2


# RUN SCENARIOS

results = [
    run_simulation(patients_normal, consult_time_map, BASE_DOCTORS, "Normal Load"),
    run_simulation(patients_peak, consult_time_map, BASE_DOCTORS, "Peak Load"),
    run_simulation(patients_peak, consult_time_map, IMPROVED_DOCTORS, "Improved Peak Load")
]


# OUTPUT RESULTS

print("\n===== HEALTHCARE SIMULATION RESULTS =====")

for r in results:
    print(f"\nScenario: {r['label']}")
    print("Doctors:", r["doctors"])
    print("Patients Arrived:", r["patients_arrived"])
    print("Patients Served:", r["patients_served"])
    print(f"Average Waiting Time: {r['avg_wait']:.2f} mins")
    print(f"Minimum Waiting Time: {r['min_wait']:.2f} mins")
    print(f"Maximum Waiting Time: {r['max_wait']:.2f} mins")
    print(f"Doctor Utilization: {r['utilization']*100:.1f}%")


# VISUALIZATIONS

output_dir = Path(r"E:\New Volume(D)\OUSL 3\EEX 5362 Performance Modelling\MP\622512695_Final_Simulation\graphs"
)
output_dir.mkdir(exist_ok=True)

for r in results:
    plt.figure()
    plt.plot(r["queue_time"], r["queue_length"])
    plt.title(f"Queue Length Over Time – {r['label']}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Patients in Queue")
    plt.tight_layout()
    plt.savefig(output_dir / f"queue_{r['label'].replace(' ', '_')}.png")
    plt.close()

plt.figure()
plt.bar([r["label"] for r in results],
        [r["utilization"] * 100 for r in results])
plt.ylabel("Doctor Utilization (%)")
plt.title("Doctor Utilization Comparison")
plt.tight_layout()
plt.savefig(output_dir / "doctor_utilization.png")
plt.close()

print("\nGraphs saved in 'graphs/' folder.")



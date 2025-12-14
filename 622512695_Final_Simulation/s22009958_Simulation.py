import simpy
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# LOAD DATA

# patients = pd.read_csv("patient.csv")
# service_log = pd.read_csv("service.csv")

patients = pd.read_csv(
    r"E:\New Volume(D)\OUSL 3\EEX 5362 Performance Modelling\MP\Simulation\patient.csv"
)

service_log = pd.read_csv(
    r"E:\New Volume(D)\OUSL 3\EEX 5362 Performance Modelling\MP\Simulation\service.csv"
)


patients["arrival_time"] = pd.to_datetime(patients["arrival_time"])
service_log["service_start_time"] = pd.to_datetime(service_log["service_start_time"])
service_log["service_end_time"] = pd.to_datetime(service_log["service_end_time"])


# BUILD CONSULTATION TIME LOOKUP

consult = service_log[service_log["service_type"] == "consultation"].copy()

consult["consult_time"] = (
    consult["service_end_time"] - consult["service_start_time"]
).dt.total_seconds() / 60

# patient_id -> consultation time (minutes)
consult_time_map = dict(
    zip(consult["patient_id"], consult["consult_time"])
)


# PREPARE PATIENT ARRIVALS

patients = patients.sort_values("arrival_time")

start_time = patients["arrival_time"].min()
patients["Arrival_Min"] = (
    patients["arrival_time"] - start_time
).dt.total_seconds() / 60

SIM_TIME = int(patients["Arrival_Min"].max() + 60)


# SIMULATION PARAMETERS

num_doctors = int(input("Enter number of doctors: "))

waiting_times = []
consultation_times = []
total_times = []


# SIMPY ENVIRONMENT

env = simpy.Environment()
doctors = simpy.Resource(env, capacity=num_doctors)


# PATIENT PROCESS

def patient(env, consult_time):
    arrival = env.now

    with doctors.request() as req:
        yield req
        start = env.now

        waiting_times.append(start - arrival)
        consultation_times.append(consult_time)

        yield env.timeout(consult_time)
        total_times.append(env.now - arrival)


# HOSPITAL PROCESS

def hospital(env, data):
    for _, row in data.iterrows():
        yield env.timeout(row["Arrival_Min"] - env.now)

        # yield env.timeout(max(0, (row["Arrival_Min"] // 5) * 5 - env.now))


        # Use real consult time if exists, else default (exam-safe)
        consult_time = consult_time_map.get(row["patient_id"], 15)

        env.process(patient(env, consult_time))


# RUN SIMULATION

env.process(hospital(env, patients))
env.run(until=SIM_TIME)


# SAFE MEAN

def safe_mean(values):
    return statistics.mean(values) if values else 0


# RESULTS

print("\n--- Healthcare System Performance Results ---")
print("Number of Doctors:", num_doctors)
print("Total Patients Served:", len(total_times))
print(f"Average Waiting Time: {safe_mean(waiting_times):.2f} mins")
print(f"Average Consultation Time: {safe_mean(consultation_times):.2f} mins")
print(f"Average Total Time in System: {safe_mean(total_times):.2f} mins")


# SIMPLE VISUALIZATION

output_dir = Path("graphs")
output_dir.mkdir(exist_ok=True)

plt.figure()
plt.plot(waiting_times)
plt.title("Patient Waiting Time")
plt.xlabel("Patient Index")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig(output_dir / "waiting_time.png")
plt.close()

print("\nGraph saved to graphs/waiting_time.png")


# VISUALIZATIONS

output_dir = Path("graphs")
output_dir.mkdir(exist_ok=True)


# LINE GRAPH – Waiting Time

if waiting_times:
    plt.figure()
    plt.plot(waiting_times, marker='o', linestyle='-')
    plt.title("Patient Waiting Time (Line Graph)")
    plt.xlabel("Patient Index")
    plt.ylabel("Waiting Time (minutes)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "waiting_time_line.png")
    plt.close()


# BAR CHART – Patients per Arrival Hour

patients["Arrival_Hour"] = (patients["Arrival_Min"] // 60).astype(int)
hour_counts = patients["Arrival_Hour"].value_counts().sort_index()

plt.figure()
plt.bar(hour_counts.index, hour_counts.values)
plt.title("Patients per Arrival Hour (Bar Chart)")
plt.xlabel("Hour")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(output_dir / "patients_per_hour_bar.png")
plt.close()


# HISTOGRAM – Consultation Duration

if consultation_times:
    plt.figure()
    plt.hist(consultation_times, bins=15, edgecolor='black')
    plt.title("Consultation Time Distribution (Histogram)")
    plt.xlabel("Consultation Time (minutes)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "consultation_time_hist.png")
    plt.close()


# SHOW SAVED FILES

print("\nSaved graph files:")
for f in output_dir.glob("*.png"):
    print("-", f)


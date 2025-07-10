# Data Requirements for Digital Twin Simulation (Idea 1)

This document outlines the data needed to build a credible simulation for the "Digital Twin Framework for Reconfigurable EV Charging Systems" research paper. The goal is to parameterize a simulation environment, not to find a single, pre-existing dataset.

## 1. Vehicle Specifications (AGVs / Electric Forklifts)

This data is required to model the physical behavior and constraints of the electric vehicles in the simulation. It is typically publicly available from manufacturer websites.

*   **Required Parameters:**
    *   **Battery Capacity:** The total energy storage of the vehicle (e.g., in kWh).
    *   **Charging Power:** The rate at which the battery can be charged (e.g., in kW). Note both AC (slower) and DC Fast Charging (if applicable) rates.
    *   **Energy Consumption Rate:** The average energy used per unit of distance or time (e.g., kWh/km or kWh/hour of operation).
    *   **Travel Speed:** The average operational speed of the vehicle.

*   **Potential Sources:**
    *   Search for "technical data sheets" or "specifications" for commercial models.
    *   **KUKA:** A major robotics and AGV manufacturer.
    *   **Omron Automation:** Provider of mobile robots.
    *   **Toyota Material Handling:** Leading manufacturer of electric forklifts.
    *   **Google Search Terms:** `"AGV technical specifications PDF"`, `"electric forklift battery capacity kWh"`.

## 2. Manufacturing Layout and Workload

This data defines the operational environment and the tasks that the vehicles must perform. This will likely be synthetically generated but based on realistic, standard benchmarks from academic literature.

*   **Required Parameters:**
    *   **Facility Layout:** A logical or physical map of the manufacturing facility, including key locations (machines, depots, warehouses, charging stations). A grid-based model (e.g., 50x50 grid) is a common and effective abstraction.
    *   **Task List (Workload):** A sequence of tasks that drives the simulation. Each task should have a start time, origin, destination, and type. Example: `(Timestamp, Task_ID, Origin, Destination)`.

*   **Potential Sources / Methodology:**
    *   **Base Layout on Academic Benchmarks:** Replicate a standard facility layout described in manufacturing systems literature. Search for papers on "job shop layout," "AGV simulation," or "factory logistics."
    *   **Synthetically Generate Workload:** Create a script to generate a random but plausible sequence of tasks. The arrival rate of tasks can be modeled using a statistical distribution (e.g., a Poisson process).
    *   **Reference Libraries:** Explore libraries like **MMSPLib (Manufacturing and Material Science Problem Library)** to understand the structure and complexity of standard benchmark problems, which you can then mimic.

## 3. Electricity Price Data

This data is crucial for demonstrating the economic benefits (cost savings) of the intelligent charging algorithm. It should reflect real-world, time-varying energy costs.

*   **Required Parameters:**
    *   **Time-Series of Electricity Prices:** A list of electricity prices ($/kWh) at regular intervals (e.g., hourly) over a representative period (e.g., a week or a month).

*   **Potential Sources:**
    *   **Grid Operator Data (Most Common in Research):**
        *   **CAISO (California ISO):** Provides historical hourly Locational Marginal Pricing (LMP) data. Search for `"CAISO OASIS LMP data"`. This is a very common source for academic papers.
        *   **PJM (Pennsylvania-New Jersey-Maryland Interconnection):** Another major US grid operator with publicly available historical price data.
    *   **Utility Rate Plans:**
        *   Search for "commercial time-of-use (ToU) electricity rates" from major utility providers. These provide a simpler but still effective price signal.

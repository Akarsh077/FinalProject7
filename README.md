# Pothole Detection System using YOLOv4 tiny and alert System  
## *Team Members:- Akarsh Singh *(leader)*, Dhruv Rai, Rajan Sharma, Shalini Tiwari.


### **The Problem Statement**

* **Hazardous Roads:** Potholes are a major safety hazard, leading to vehicle damage, traffic accidents, and severe injuries.
* **Slow, Manual Monitoring:** Traditional road maintenance relies on manual inspections or citizen complaints, which are incredibly slow and inefficient.
* **Heavy Hardware Requirements:** Standard deep learning models are too computationally heavy to run in real-time on budget-friendly dashboard cameras or edge devices.

### **The Solution**

* **Lightweight Real-Time Detection:** This system uses **YOLOv4-tiny**, an optimized, high-speed neural network designed to detect potholes in real-time on low-power devices.
* **Automated Data Logging:** As a vehicle drives, the system instantly processes the camera feed, detects potholes, and saves their precise coordinates directly to a `pothole_data.csv` file.
* **Interactive Mapping:** The logged coordinates are plotted onto an interactive map (`pothole_map.html`) to visualize the worst-hit areas.
* **Proactive Maintenance:** This provides municipal authorities with a highly automated, low-cost tool to map road damage instantly and dispatch repair crews efficiently.

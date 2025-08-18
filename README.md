# 🌍 Climate in the Light of Mathematical Equations 
**ECMI Modeling Week 2025**  
**Kaunas University of Technology, Lithuania**  
**Date:** July 5, 2025  

**Contributors:**  
- Dr. Davor Kumozec (University of Novi Sad)  
- Valeriia Baranivska (Igor Sikorsky Kyiv Polytechnic Institute)  
- Ilaria Astrid Bartsch (Università degli Studi di Milano)  
- Janne Finn Heibel (University of Koblenz)  
- Patrícia Marques (Instituto Superior Técnico)  

---

## 📘 Overview

This project explores two complementary approaches to modeling atmospheric dynamics and climate behavior:

1. **Numerical Simulation via PDEs**  
2. **Statistical Learning using a Hybrid SARIMAX–LSTM Model**

Our goal is to understand the dynamics of a one-dimensional fluid system influenced by solar heating, simulating day-night cycles and their impact on atmospheric variables such as pressure, temperature, and velocity.

---

## 🧮 Methodologies

### 1. PDE-Based Modeling

We simulate atmospheric behavior using a system of partial differential equations (PDEs) derived from the **Euler equations**. These equations are closed with the **ideal gas law** and numerically solved using the **Lax–Friedrichs scheme** after discretization.

### 2. Hybrid Machine Learning Model

We implement a **hybrid forecasting model** that combines **SARIMAX** (Seasonal Autoregressive Integrated Moving Average with exogenous regressors) with **LSTM** (Long Short-Term Memory Neural Network). This hybrid approach models time series behavior of atmospheric variables, learning from both deterministic patterns and complex, non-linear relationships.

---

# Geekathon25

Urban intersections are major sources of traffic congestion, fuel consumption, and vehicle emissions. Traditional traffic lights often operate on fixed schedules that do not adapt to fluctuating traffic patterns. As a result, vehicles spend excessive time idling or accelerating repeatedly, which increases travel time, fuel use, and air pollution.

When traffic lights are poorly timed, misconfigured, or malfunctioning, these inefficiencies are amplified, creating bottlenecks and higher emissions. Conventional control methods lack the ability to learn and adapt dynamically, which limits their effectiveness in real-world, ever-changing traffic conditions.

This project implements a reinforcement learning (RL) approach to traffic light control, specifically using Q-learning. RL enables the traffic lights to learn optimal signal timings based on the current traffic state. Each traffic light “agent” receives feedback from its environment—such as vehicle waiting times or queue lengths—and adjusts its behavior to maximize a defined reward, in this case minimizing stopped time and emissions.

By comparing a baseline simulation using real-world fixed traffic light schedules with the RL-controlled system, we demonstrate that adaptive, self-learning traffic lights can improve traffic flow, reduce vehicle idling, and lower pollution at intersections.


Future Roadmap
- [ ] Scaling Up: Extend the RL system to manage multiple intersections in a coordinated network rather than a single crossing.
- [ ] Multi-Objective Optimization: Incorporate additional metrics into the reward function, such as fuel consumption, CO₂ emissions, and pedestrian safety.
- [ ] Robustness & Deployment: Test the RL system under extreme or unpredictable traffic scenarios and explore integration with city traffic management platforms.
- [ ] Emergency Vehicle Priority: Adapt the RL system to detect and prioritize ambulances, fire trucks, and other emergency vehicles, ensuring minimal delay during critical situations.

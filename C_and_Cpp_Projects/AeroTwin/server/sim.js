/**
 * AeroTwin - High-Fidelity Dynamic Digital Twin
 * 
 * CORE PHYSICS UPGRADE:
 * The simulation now uses Forces (Lift, Weight, Drag) to determine Z-acceleration.
 * Discovery Delta: Difference between Predicted Aero State and Actual Aero State.
 * Flux Residual: The "Physics-Violation" metric (Error in the Inverse solver).
 */

class Vector3 {
    constructor(x = 0, y = 0, z = 0) {
        this.x = x; this.y = y; this.z = z;
    }
    add(v) { return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z); }
    sub(v) { return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z); }
    multiply(s) { return new Vector3(this.x * s, this.y * s, this.z * s); }
    norm() { return Math.sqrt(this.x ** 2 + this.y ** 2 + this.z ** 2); }
    clone() { return new Vector3(this.x, this.y, this.z); }
}

class Quaternion {
    constructor(w = 1, x = 0, y = 0, z = 0) {
        this.w = w; this.x = x; this.y = y; this.z = z;
    }
    normalize() {
        const mag = Math.sqrt(this.w ** 2 + this.x ** 2 + this.y ** 2 + this.z ** 2);
        if (mag > 0) { this.w /= mag; this.x /= mag; this.y /= mag; this.z /= mag; }
        return this;
    }
    multiply(q) {
        return new Quaternion(
            this.w * q.w - this.x * q.x - this.y * q.y - this.z * q.z,
            this.w * q.x + this.x * q.w + this.y * q.z - this.z * q.y,
            this.w * q.y - this.x * q.z + this.y * q.w + this.z * q.x,
            this.w * q.z + this.x * q.y - this.y * q.x + this.z * q.w
        );
    }
}

class AeroTwinSimulator {
    constructor() {
        this.params = {
            mass: 1.5,
            g: 9.81,
            rho: 1.225,
            S: 0.12,
            CL_slope: 5.2, // Coefficient of Lift slope
            noise: 0.02
        };

        // RLS Estimatation State
        this.rls = {
            theta: 0.1,  // Current identified CL
            P: 100.0,    // Covariance
            lambda: 0.98 // Forgetting factor
        };

        this.reset();
    }

    reset() {
        this.state = {
            pos: new Vector3(0, 0, 0),
            vel: new Vector3(0.1, 0, 0),
            acc: new Vector3(0, 0, 0),
            orient: new Quaternion(1, 0, 0, 0),
            trueCL: 0.2, // Hidden variable
            aoa: 0
        };
        this.totalTime = 0;
    }

    step(dt, pathType = 'circle') {
        this.totalTime += dt;

        // --- 1. RESEARCH TRAJECTORY (GUIDANCE) ---
        let targetPos = new Vector3(0, 0, -50.0);
        const t = this.totalTime;
        if (pathType === 'square') {
            const side = 150.0, lap = 40.0;
            const s = (t % lap) / lap * 4;
            const p = (t % (lap / 4)) / (lap / 4) * side;
            if (s < 1) targetPos = new Vector3(p, 0, -50);
            else if (s < 2) targetPos = new Vector3(side, p, -50);
            else if (s < 3) targetPos = new Vector3(side - p, side, -50);
            else targetPos = new Vector3(0, side - p, -50);
        } else if (pathType === 'climb') {
            targetPos = new Vector3(20 * t, 0, -50 - 5 * t);
        } else {
            const r = 80, w = 0.25;
            targetPos = new Vector3(r * Math.cos(w * t), r * Math.sin(w * t), -50);
        }

        // --- 2. DYNAMIC PHYSICS ENGINE ---
        // We simulate the flight using actual Aero Forces
        const dir = targetPos.sub(this.state.pos);
        const dist = dir.norm();
        const v_mag = this.state.vel.norm();

        // Aerodynamic State
        // AoA = angle between velocity and horizon
        this.state.aoa = Math.atan2(-this.state.vel.z, Math.sqrt(this.state.vel.x ** 2 + this.state.vel.y ** 2)) || 0;
        this.state.trueCL = Math.abs(this.params.CL_slope * this.state.aoa) + 0.1; // "The Ground Truth"

        // Calculate Forces
        const dynam_p = 0.5 * this.params.rho * (v_mag ** 2) * this.params.S;
        const lift_force = dynam_p * (this.state.trueCL + (Math.random() - 0.5) * this.params.noise);

        // Integrated 6-DOF Dynamics (Simplifed for Z-Physics)
        const oldVel = this.state.vel.clone();

        // Lateral movement is guidance-controlled (High performance controller)
        const v_lat_target = (dist > 1) ? dir.multiply(1 / dist).multiply(22) : new Vector3(0, 0, 0);
        this.state.vel.x += (v_lat_target.x - this.state.vel.x) * 0.8 * dt;
        this.state.vel.y += (v_lat_target.y - this.state.vel.y) * 0.8 * dt;

        // Vertical movement is FORCE-DRIVEN (Aero + Gravity)
        // a_z = (Weight - Lift) / m (Weight is positive down in NED)
        const acc_z = (this.params.mass * this.params.g - lift_force) / this.params.mass;
        this.state.vel.z += acc_z * dt;

        this.state.pos = this.state.pos.add(this.state.vel.multiply(dt));
        this.state.acc = this.state.vel.sub(oldVel).multiply(1 / dt);

        // --- 3. INVERSE IDENTIFICATION ENGINE ---
        // This machine tries to figure out what 'trueCL' is just by watching pos/vel/acc
        if (v_mag > 5.0) {
            // Observed Lift Force from motion measurements
            // F = m * g - m * acc_z
            const F_obs = this.params.mass * (this.params.g - this.state.acc.z);
            const phi = dynam_p;
            const y = F_obs;

            // RLS Step
            const K = this.rls.P * phi / (this.rls.lambda + phi * this.rls.P * phi);
            this.rls.theta = this.rls.theta + K * (y - phi * this.rls.theta);
            this.rls.P = (this.rls.P - K * phi * this.rls.P) / this.rls.lambda;
        }

        // --- 4. METRICS CALCULATION ---
        // Flux Residual: How much the identified model fails to predict the observation
        const predicted_F = dynam_p * this.rls.theta;
        const flux_res = Math.abs(predicted_F - (this.params.mass * (this.params.g - this.state.acc.z))) / (this.params.mass * this.params.g + 1);

        // Discovery Delta: The actual error in our identified Aero property
        const discovery_delta = Math.abs(this.rls.theta - this.state.trueCL);

        // Orientation
        const yaw = Math.atan2(this.state.vel.y, this.state.vel.x) || 0;
        this.state.orient = new Quaternion(Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)).normalize();

        return {
            t: this.totalTime,
            px: this.state.pos.x, py: this.state.pos.y, pz: this.state.pos.z,
            alt: -this.state.pos.z,
            vx: v_mag,
            qw: this.state.orient.w, qx: this.state.orient.x, qy: this.state.orient.y, qz: this.state.orient.z,
            cl: parseFloat(this.rls.theta.toFixed(4)),
            cl_true: parseFloat(this.state.trueCL.toFixed(4)),
            aoa: (this.state.aoa * 180 / Math.PI).toFixed(2),
            res: parseFloat(flux_res.toFixed(6)),
            delta: parseFloat(discovery_delta.toFixed(4)),
            wp: pathType.toUpperCase()
        };
    }
}

module.exports = AeroTwinSimulator;

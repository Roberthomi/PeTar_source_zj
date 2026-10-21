#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <type_traits>
#include <vector>
#define ASSERT assert
#define DATADUMP(x) abort()
#include "AR/symplectic_integrator.h"
#include "Hermite/ar_information.h"
#include "Hermite/block_time_step.h"
#include "Hermite/neighbor.h"
#include "Hermite/profile.h"
// Expose internal slot lists only in this regression test.
#define private public
#include "Hermite/hermite_integrator.h"
#undef private
#include "../sample/Hermite/particle.h"
#include "../sample/Hermite/hermite_perturber.h"
#include "../sample/Hermite/ar_interaction.h"
#include "../sample/Hermite/hermite_interaction.h"
#include "../sample/Hermite/hermite_information.h"

using Integrator = H4::HermiteIntegrator<Particle, Particle, HermitePerturber,
    H4::Neighbor<Particle>, HermiteInteraction, ARInteraction, HermiteInformation>;

static void run(bool merger) {
    Integrator h;
    std::remove_pointer<decltype(h.ar_manager)>::type manager;
    h.ar_manager = &manager;
    manager.interaction.gravitational_constant = 1.0;
    h.groups.setMode(COMM::ListMode::local);
    h.groups.reserveMem(3);
    h.groups.resizeNoInitialize(3);
    h.table_group_mask_.setMode(COMM::ListMode::local);
    h.table_group_mask_.reserveMem(3);
    for (int k=0; k<3; ++k) h.table_group_mask_.addMember(k==1);
    h.index_dt_sorted_group_.setMode(COMM::ListMode::local);
    h.index_dt_sorted_group_.reserveMem(2);
    h.index_group_merger_.setMode(COMM::ListMode::local);
    h.index_group_merger_.reserveMem(1);
    h.index_offset_group_ = 10;
    for (int k : {0, 2}) {
        h.index_dt_sorted_group_.addMember(k);
        auto& g = h.groups[k];
        g.particles.setMode(COMM::ListMode::local);
        g.particles.reserveMem(2);
        Particle a, b;
        a.mass = b.mass = 1.0;
        a.id = 1; b.id = 2;
        b.pos[0] = 1.0;
        b.vel[0] = 3.0; // Outgoing hyperbolic pair: must break.
        g.particles.addMember(a);
        g.particles.addMember(b);
        g.info.reserveMem(2);
        g.info.r_break_crit = 0.1;
    }
    if (merger) h.index_group_merger_.addMember(2);
    int output[4] = {}, count = 0;
    h.checkBreak(output, count, true);
    assert(count == 2);
    assert(output[0] == (merger ? 12 : 10));
    assert(output[1] == (merger ? 10 : 12));
    assert(h.index_group_merger_.getSize() == 0);
}

static void checkInitialMemberNeighbors() {
    Integrator h;
    std::remove_pointer<decltype(h.manager)>::type hm;
    std::remove_pointer<decltype(h.ar_manager)>::type am;
    h.manager = &hm;
    h.ar_manager = &am;
    hm.interaction.eps_sq = am.interaction.eps_sq = 0.0;
    hm.interaction.gravitational_constant = am.interaction.gravitational_constant = 1.0;
    hm.step.eta_2nd = hm.step.eta_4th = 0.1;
    hm.step.acc0_offset_sq = 0.0;
    am.time_error_max = am.energy_error_relative_max = 1.e-8;
    am.time_step_min = 1.e-10;
    am.slowdown_pert_ratio_ref = 1.e-6;
    am.slowdown_timescale_max = 1.0;
    am.step_count_max = 1000;
    am.step.initialSymplecticCofficients(2);
    Particle::r_neighbor_crit = 3.0;
    Particle::r_break_crit = 0.1;
    h.particles.setMode(COMM::ListMode::local);
    h.particles.reserveMem(3);
    for (int i=0; i<3; ++i) {
        Particle p;
        p.id = i+1; p.mass = 1.0;
        p.pos[0] = i; p.vel[1] = 0.1*i;
        h.particles.addMember(p);
    }
    h.groups.setMode(COMM::ListMode::local);
    h.groups.reserveMem(1);
    h.reserveIntegratorMem();
    for (int i=0; i<3; ++i) {
        auto& nb = h.neighbors.getDataAddress()[i];
        nb.r_neighbor_crit_sq = 100.0;
        nb.initial_step_flag = true;
    }
    h.initialSystemSingle(0.0);
    // Empty requests and a nonempty request with no group slots are valid.
    const Float original_x = h.particles[0].pos[0];
    h.integrateToTimeList(0.0, nullptr, 0);
    h.particles[0].time = 0.0;
    const int single[] = {0};
    h.integrateToTimeList(0.0, single, 1);
    assert(h.particles[0].pos[0] == original_x);
    assert(h.groups.getSize() == 0);
    int members[] = {0, 1}, offsets[] = {0, 2};
    h.addGroups(members, offsets, 1);
    assert(h.n_init_single_ == 1);
    for (int i=0; i<3; ++i) {
        assert(h.neighbors[i].r_neighbor_crit_sq == 9.0);
        assert(!h.neighbors[i].initial_step_flag);
    }
    // A group member's radius must admit a neighbor outside the other's radius.
    h.neighbors[2].r_neighbor_crit_sq = 1.0;
    h.neighbors[0].resetNeighbor();
    h.neighbors[0].checkAndAddNeighborSingle(4.0, h.particles[2], h.neighbors[2], 2);
    assert(h.neighbors[0].n_neighbor_single == 1);
}

int main() {
    checkInitialMemberNeighbors();
    run(false); // Sparse active slot read.
    run(true);  // Sparse merger slot write and no duplicate break.
    std::cout << "group break mask regression passed\n";
}

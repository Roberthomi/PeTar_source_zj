#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#define ASSERT assert
#include "Hermite/neighbor.h"

struct Particle { Float mass = 1.0; };
using Neighbor = H4::Neighbor<Particle>;
struct Group {
    COMM::ParticleGroup<Particle, H4::ParticleH4<Particle>> particles;
    Neighbor perturber;
};

static void checkSelection(Neighbor& nb) {
    assert(nb.r_min_index == -1 && nb.r_min_mass == 0.0);
    nb.r_neighbor_crit_sq = 100.0;
    Neighbor other;
    H4::ParticleH4<Particle> p;
    p.mass = 0.0;
    nb.checkAndAddNeighborSingle(1.0, p, other, 0);
    assert(nb.r_min_index == -1);
    p.mass = 2.0;
    nb.checkAndAddNeighborSingle(8.0, p, other, 1);
    assert(nb.r_min_index == 1 && nb.r_min_mass == 2.0);
    Group group;
    group.particles.cm.mass = 8.0;
    nb.checkAndAddNeighborGroup(16.0, group, 2);
    assert(nb.r_min_index == 2 && nb.r_min_mass == 8.0);
    p.mass = 1.0;
    nb.checkAndAddNeighborSingle(3.0, p, other, 3);
    assert(nb.r_min_index == 2); // Closer geometrically, but weaker mass weighting.
    p.mass = 4.0;
    nb.checkAndAddNeighborSingle(4.0, p, other, 4);
    assert(nb.r_min_index == 4);
}

int main() {
    Neighbor nb;
    nb.reserveMem(64);
    checkSelection(nb);
    nb.resetNeighbor();
    checkSelection(nb);
    nb.clearNoFreeMemNoResizeNeighborAdress();
    checkSelection(nb);
    nb.clearNoFreeMem();
    checkSelection(nb);
    nb.clear();
    nb.reserveMem(64);
    checkSelection(nb);
    nb.resetNeighbor();
    Group group;
    group.particles.cm.mass = 0.0;
    nb.checkAndAddNeighborGroup(1.0, group, 5);
    assert(nb.r_min_index == -1);
    group.particles.cm.mass = 3.0;
    nb.checkAndAddNeighborGroup(9.0, group, 6);
    assert(nb.r_min_index == 6 && nb.r_min_mass == 3.0);
    std::cout << "neighbor initialization regression passed\n";
}

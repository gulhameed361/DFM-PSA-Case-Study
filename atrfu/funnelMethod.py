# -*- coding: utf-8 -*-
"""
Funnel globalisation for the trust-region method (strategy 1).

The funnel keeps one scalar width phi that bounds the infeasibility theta of
the iterates and shrinks as feasibility improves, plus the best objective
value seen. A trial point with infeasibility theta+ and objective f+, from an
iterate with theta_k and f_k, is classified as

  inside the funnel (theta+ <= phi):
    f-type      if  f_k - f+ >= mu_s * theta_k^p          (switching)
                and f_k - f+ >= eta * Delta_k             (Armijo)
                -- rejected if switching holds but Armijo fails;
    theta-type  if switching fails and theta+ <= beta * phi;
  outside the funnel:
    theta-type  (relaxed) if switching holds and theta+ <= kappa_r * phi;
  otherwise rejected.

After a theta-type step the funnel contracts,
phi <- max(phi_min, (1 - kappa_f) theta+ + kappa_f phi).

@author: Gul Hameed
"""


class Funnel:
    """Scalar funnel: width phi and best objective f_best.

    Parameters
    ----------
    phi_init, f_best_init : initial width (the initial infeasibility) and objective
    phi_min               : floor on the width
    kappa_f               : contraction weight after a theta-type step
    beta                  : required shrink for a theta-type step inside the funnel
    switch_coeff          : switching coefficient mu_s
    switch_exponent       : switching exponent p
    eta                   : Armijo coefficient
    kappa_r               : width factor for a theta-type step outside the funnel
    """

    def __init__(self, phi_init, f_best_init, phi_min, kappa_f, beta,
                 switch_coeff, switch_exponent, eta, kappa_r):
        self.phi = max(phi_min, phi_init)
        self.f_best = f_best_init
        self.phi_min = phi_min
        self.kappa_f = kappa_f
        self.beta = beta
        self.switch_coeff = switch_coeff
        self.switch_exponent = switch_exponent
        self.eta = eta
        self.kappa_r = kappa_r

    def _switching(self, f_old, f_new, theta_old):
        return (f_old - f_new) >= self.switch_coeff * (theta_old ** self.switch_exponent)

    def _armijo(self, f_old, f_new, delta):
        return (f_old - f_new) >= self.eta * delta

    def classify_step(self, theta_old, theta_new, f_old, f_new, delta):
        """Return 'f', 'theta', 'theta-relax' or 'reject' for the trial point."""
        if theta_new <= self.phi:
            if self._switching(f_old, f_new, theta_old):
                return 'f' if self._armijo(f_old, f_new, delta) else 'reject'
            return 'theta' if theta_new <= self.beta * self.phi else 'reject'
        if (self._switching(f_old, f_new, theta_old)
                and theta_new <= self.kappa_r * self.phi):
            return 'theta-relax'
        return 'reject'

    def accept_f(self, theta_new, f_new):
        """Record an accepted f-type step."""
        if f_new < self.f_best:
            self.f_best = f_new

    def accept_theta(self, theta_new):
        """Contract the funnel after an accepted theta-type step."""
        self.phi = max(self.phi_min,
                       (1 - self.kappa_f) * theta_new + self.kappa_f * self.phi)

    def relax_theta(self, theta_new):
        """Contract the funnel after a theta-type step taken outside it."""
        self.phi = max(self.phi_min,
                       (1 - self.kappa_f) * theta_new + self.kappa_f * self.phi)

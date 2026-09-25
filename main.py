# Piešķiršanas problēma, risināta ar simulated annealing.

# Domēns - permutācijas: assignment[i] ir darbs, ko veic darbinieks i.
# Izmaksu funkcija - summa cost_matrix[i][assignment[i]] pa visiem darbiniekiem.
# Apkārtne - divu darbinieku darbu samainīšana.

import numpy as np

FORBIDDEN_COST = 10_000.0  # darbinieks šo darbu nevar veikt
FORBIDDEN_RATIO = 0.15  # cik liela daļa no pāriem (darbinieks, darbs) ir aizliegti


def generate_instance(size, seed):
    # Izmaksas = stundas likme * darba laiks + sods par nepietiekamu kvalifikāciju.
    rng = np.random.default_rng(seed)
    shape = (size, size)

    hourly_wage = rng.uniform(10, 30, size)  # eur/h
    base_job_hours = rng.uniform(5, 20, size)  # stundas
    skill_factor = rng.uniform(0.5, 2.0, shape)  # >1 nozīmē, ka darbiniekam i darbs j aizņem ilgāk
    work_hours = base_job_hours[None, :] * skill_factor
    qualification_penalty = np.where(rng.random(shape) < 0.2, rng.uniform(20, 80, shape), 0.0)
    cost_matrix = hourly_wage[:, None] * work_hours + qualification_penalty

    is_forbidden = rng.random(shape) < FORBIDDEN_RATIO
    # lai vienmēr būtu vismaz viens derīgs risinājums
    guaranteed_assignment = rng.permutation(size)
    is_forbidden[np.arange(size), guaranteed_assignment] = False
    cost_matrix[is_forbidden] = FORBIDDEN_COST
    return cost_matrix


def run_benchmark():
    pass


if __name__ == "__main__":
    run_benchmark()

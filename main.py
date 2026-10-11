# Piešķiršanas problēma, risināta ar simulated annealing.

# Domēns - permutācijas: assignment[i] ir darbs, ko veic darbinieks i.
# Izmaksu funkcija - summa cost_matrix[i][assignment[i]] pa visiem darbiniekiem.
# Apkārtne - divu darbinieku darbu samainīšana.

# scipy linear_sum_assignment izmanto tikai, lai salīdzinātu ar optimālo atbildi.

# Izmantotā literatūra:
# https://www.geeksforgeeks.org/artificial-intelligence/what-is-simulated-annealing/

# Matīss Aizgalis, ma26049

import math
import random
import time
from pathlib import Path
from statistics import mean
import numpy as np
from scipy.optimize import linear_sum_assignment

FORBIDDEN_COST = 10_000.0  # darbinieks šo darbu nevar veikt
FORBIDDEN_RATIO = 0.15  # cik liela daļa no pāriem (darbinieks, darbs) ir aizliegti

COOLING_RATE = 0.98  # alfa, T = T * alfa
MIN_TEMPERATURE = 1e-8  # apstājamies, kad T < MIN_TEMPERATURE (izvēlētā vērtība no izvēlētās literatūras piemēra)
ITERATIONS_PER_SIZE = 50  # kaimiņu skaits pie katras temperatūras = 50 * N

BENCHMARK_SIZES = (10, 20, 50, 100)
BENCHMARK_RUNS = 5  # cik reizes palaižam SA katram izmēram

INSTANCES_DIR = Path(__file__).parent / "instances"

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

def assigned_costs(cost_matrix, assignment):
    return cost_matrix[np.arange(len(assignment)), assignment]

def total_cost(cost_matrix, assignment):
    return float(assigned_costs(cost_matrix, assignment).sum())

def swap_cost_change(cost_matrix, assignment, worker_a, worker_b):
    # skatāmies tikai uz divām mainītajām rindām, nevis pārrēķinām visu

    job_a, job_b = assignment[worker_a], assignment[worker_b]
    cost_after = cost_matrix[worker_a, job_b] + cost_matrix[worker_b, job_a]
    cost_before = cost_matrix[worker_a, job_a] + cost_matrix[worker_b, job_b]
    return cost_after - cost_before

def is_valid_assignment(cost_matrix, assignment):
    is_permutation = sorted(assignment) == list(range(len(assignment)))
    return is_permutation and bool((assigned_costs(cost_matrix, assignment) < FORBIDDEN_COST).all())

def simulated_annealing(cost_matrix, seed):
    # Sākam ar augstu temperatūru un nejaušu risinājumu. Katrā solī paņemam kaimiņu
    # (samainām divu darbinieku darbus).
    # Labāku kaimiņu pieņemam vienmēr, sliktāku - ar varbūtību e^(-delta/T) (Lai tiktu laukā no lokālā minimuma).
    # Temperatūrai krītot, sliktākus gājienus pieņemam arvien retāk, un meklēšana nostabilizējas.

    rng = random.Random(seed)
    size = len(cost_matrix)
    # cik kaimiņus apskatām pie katras temperatūras
    iterations_per_temp = ITERATIONS_PER_SIZE * size

    # sākuma temperatūra - izmaksu standartnovirze, lai sākumā e^(-delta/T)
    # būtu pietiekami liels un algoritms varētu brīvi pētīt risinājumu telpu
    temperature = float(cost_matrix[cost_matrix < FORBIDDEN_COST].std())

    # sākuma risinājums - nejauša permutācija
    assignment = list(range(size))
    rng.shuffle(assignment)
    current_cost = total_cost(cost_matrix, assignment)
    # atceramies labāko atrasto, jo pašreizējais var arī pasliktināties
    best_assignment, best_cost = assignment[:], current_cost

    # apstāšanās kritērijs - kad temperatūra kļuvusi pietiekami zema
    while temperature > MIN_TEMPERATURE:
        for _ in range(iterations_per_temp):
            # kaimiņš: izvēlamies divus darbiniekus, kuriem samainīt darbus
            worker_a, worker_b = rng.sample(range(size), 2)
            cost_change = swap_cost_change(cost_matrix, assignment, worker_a, worker_b)

            # labāku pieņemam vienmēr, sliktāku ar varbūtību P = e^(-delta/T)
            # (jo augstāka T un mazāka pasliktināšanās, jo lielāka P)
            accept = cost_change < 0 or rng.random() < math.exp(-cost_change / temperature)
            if not accept:
                continue

            assignment[worker_a], assignment[worker_b] = assignment[worker_b], assignment[worker_a]
            current_cost += cost_change
            if current_cost < best_cost:
                best_assignment, best_cost = assignment[:], current_cost

        # atdzesēšana: T = T * alfa
        temperature *= COOLING_RATE

    return best_assignment, best_cost

def optimal_cost_reference(cost_matrix):
    workers, jobs = linear_sum_assignment(cost_matrix)
    return float(cost_matrix[workers, jobs].sum())

def load_instance(size):
    # testēšanas piemēri glabājas instances/inst_N.csv
    # tiek uzģenerēti un saglabāti faili
    path = INSTANCES_DIR / f"inst_{size}.csv"
    if not path.exists():
        INSTANCES_DIR.mkdir(exist_ok=True)
        np.savetxt(path, generate_instance(size, seed=size), delimiter=",", fmt="%.2f")
    return np.loadtxt(path, delimiter=",")

def run_benchmark():
    print(f"{'N':>5} {'optimums':>11} {'SA vid.':>11} {'SA labākais':>12} {'novirze %':>10} {'trāpīja opt.':>13} {'laiks (s)':>10}")

    for size in BENCHMARK_SIZES:
        cost_matrix = load_instance(size)
        optimal_cost = optimal_cost_reference(cost_matrix)

        run_costs, run_durations = [], []
        for run_index in range(BENCHMARK_RUNS):
            start_time = time.perf_counter()
            assignment, found_cost = simulated_annealing(cost_matrix, seed=run_index)
            run_durations.append(time.perf_counter() - start_time)

            assert is_valid_assignment(cost_matrix, assignment), "Nederīgs risinājums!"
            run_costs.append(found_cost)

        average_cost = mean(run_costs)
        # max ar 0, lai noapaļošanas kļūdas dēļ neparādās -0.00
        gap_percent = max(100 * (average_cost - optimal_cost) / optimal_cost, 0.0)
        optimal_hits = sum(math.isclose(cost, optimal_cost, abs_tol=1e-6) for cost in run_costs)

        print(f"{size:>5} {optimal_cost:>11.2f} {average_cost:>11.2f} {min(run_costs):>12.2f} "
              f"{gap_percent:>10.2f} {optimal_hits:>8}/{BENCHMARK_RUNS:<4} {mean(run_durations):>10.2f}")

if __name__ == "__main__":
    run_benchmark()

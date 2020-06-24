#online learning
import math
def beta_binomial(trails, a, b):
    a = int(a)
    b = int(b)
    prior_a = a
    prior_b = b
    posterior_a = a
    posterior_b = b
    for idx, trail in enumerate(trails):
        print("case " + str(idx+1)+ ": " + str(trail))
        for i in trail:
            if i == '1':
                posterior_a += 1
            else:
                posterior_b += 1
        likelihood = calculate_likelihood(posterior_a-prior_a, posterior_b-prior_b)
        print("Likelihood: " + str(likelihood))
        print("Beta prior:\ta = " + str(prior_a) + " b = " + str(prior_b))
        print("Beta posterior:\ta = " + str(posterior_a) + " b = " + str(posterior_b))
        print("\n")
        prior_a = posterior_a
        prior_b = posterior_b


def calculate_likelihood(a, b):
    likelihood = math.factorial(a+b)/math.factorial(a)/math.factorial(b)*((a/(a+b))**a)*((b/(a+b))**b)
    return likelihood
    



if __name__ == "__main__":
    a = input("Parameter a for the initial beta prior: ")
    b = input("Parameter b for the initial beta prior: ")
    trails = dict()
    f = open("testcase.txt", "r")
    if f.mode == 'r':
        trails = f.read().splitlines()
    beta_binomial(trails, a, b)
    f.close
import cvxpy as cp
import warnings
import numpy as np

class MPCController():

    def __init__(self, K, T, N, risk_free, risk, trade=0.005, risk_type='variance', cvar_alpha=None):

        risk_types = ('variance', 'semi_variance', 'cvar')
        if risk_type not in risk_types:
            raise ValueError(f'risk type must be one of {risk_types}')

        if risk_type == "cvar":
            if cvar_alpha is None:
                raise ValueError("cvar_alpha must be provided when using risk_type: cvar")
            
            tail_count = (1 - cvar_alpha) * K
            if tail_count < 30:
                    warnings.warn(
                        f"CVaR_{cvar_alpha:.2f} uses only "
                        f"{tail_count:.1f} tail scenarios. "
                        "Results may be unstable. "
                        "Consider increasing the number of scenarios."
                    )

        self.risk_type = risk_type

        # Parameters and Constants

        self.x0 = cp.Parameter(N+1, name='initial wealth')
        self.r = [cp.Parameter((K,N), name=f'returns_time_{t}') for t in range(T)]

        self.scaled_risk = cp.Parameter(nonneg=True)

        self.risk_free = cp.Constant(risk_free, name='risk free rate')

        self.trade = cp.Parameter(nonneg=True, value = trade, name="trade_cost") # value=0.005 is .5%
        self.risk = cp.Parameter(nonneg=True, value = risk, name='risk_aversion')

        # Variables

        self.u = cp.Variable((T, N+1))
        self.x = [cp.Variable((K, N+1), nonneg = True) for _ in range(T+1)]  # scenario number, time, ith asset - and no shorting

        # AUX variables
        self.z = cp.Variable() # expected final wealth
        
        # negative and positive trades
        self.pos, self.neg = cp.Variable((T, N+1), nonneg=True), cp.Variable((T, N+1), nonneg=True) 

        # Constraints
        
        constraints = [self.x[0] == self.x0]
        #constraints += [x[:, T-1, 1:] <= 1e-4] # End with cash only

        constraints.append(self.pos - self.neg == self.u)

        for t in range(T):

            #if t > 0:
            #    constraints.append(
            #        cp.sum(x[:, t, 1:], axis=1) >= 0.8 * cp.sum(x[:, t, :], axis=1) # NOTE: For testing
            #    )

            constraints.append(
                #cp.sum(self.u[t, :]) + trade*cp.norm1(self.u[t, 1:]) <= 0  # self-financing condition
                cp.sum(self.u[t, :]) + self.trade * cp.sum(self.pos[t, 1:]) + self.trade * cp.sum(self.neg[t, 1:]) == 0
            )
            
            R_t = 1 + self.r[t]
            constraints.append(
            self.x[t+1][:, 1:] == cp.multiply(self.x[t][:, 1:], R_t) + cp.multiply(self.u[t, 1:], R_t)
            )

            constraints.append(
                self.x[t+1][:, 0] == self.u[t, 0]*(1+risk_free) + self.x[t][:, 0]*(1+risk_free) # Cash evolution
            )


        final_wealth = cp.sum(self.x[T], axis=1)  # shape (K,)

        constraints.append(self.z == cp.sum(final_wealth) / K)
        
        if risk_type == 'variance':
            risk_measure = cp.var(final_wealth)

        if risk_type == 'semi_variance':
            self.downside = cp.Variable(K, nonneg=True)
            constraints += [self.downside >= self.z - final_wealth]
            risk_measure = cp.sum_squares(self.downside) / K
        
        if risk_type == "cvar":
            self.tau = cp.Variable()
            risk_measure = self.tau + 1/(1 - cvar_alpha) * cp.mean(cp.pos(-final_wealth -self.tau))

        #variance_regularization = True
        #if variance_regularization:
        #    variance = cp.var(final_wealth)

        objective = cp.Maximize(
            self.z - self.scaled_risk * risk_measure
            )

        self.problem = cp.Problem(objective, constraints)

    def is_QP_and_DPP(self):
        return self.problem.is_qp(), self.problem.is_dpp()
    
    def update_parameters(self, scenarios, x0):
        for t, r_t in enumerate(self.r):
            r_t.value = scenarios[:, t, :]
        
        self.x0.value = x0

        if self.risk_type in ('variance', 'semi_variance'):
            self.scaled_risk.value = self.risk.value / np.sum(x0)
        else:
            self.scaled_risk.value = self.risk.value


    def solve(self, solver='OSQP'):

        if solver == 'HIGHS':
            self.problem.solve(solver=cp.HIGHS, verbose=False)

        elif solver == 'CLARABEL':
            self.problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                verbose=False,
                max_iter=100,
                tol_gap_abs=1e-3,
                tol_gap_rel=1e-3,
                tol_feas=1e-3,
            )

        elif solver == 'SCS':
            self.problem.solve(
                solver=cp.SCS,
                warm_start=True,
                eps=1e-3,
                max_iters=3000,
                acceleration_lookback=10,
                verbose=False,
            )


        elif solver == 'OSQP':
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                polish=True,
                eps_abs=1e-3,
                eps_rel=1e-3,
                max_iter=3000
            )
        
        else:
            raise ValueError("Invalid solver name.")

        if self.problem.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            raise RuntimeError(f"Problem is {self.problem.status}")
        

        return {
            'objective': self.problem.value,
            'status': self.problem.status,
            'solve_time': self.problem.solver_stats.solve_time,
            'compilation_time': self.problem.compilation_time
        }
    
    def control(self):
        """Return first control action"""
        return self.u[0].value.copy()
        
    
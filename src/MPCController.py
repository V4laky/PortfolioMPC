import cvxpy as cp


class MPCController():

    def __init__(self, K, T, N, risk_free, risk, trade=0.005, risk_type='variance'):

        risk_types = ('variance', 'semi_variance')
        if risk_type not in risk_types:
            raise ValueError(f'risk type must be one of {risk_types}')

        # Parameters and Constants

        self.x0 = cp.Parameter(N+1, name='initial wealth')
        self.r = [cp.Parameter((K,N), name=f'returns_time_{t}') for t in range(T)]

        self.risk_free = cp.Constant(risk_free, name='risk free rate')

        self.trade = cp.Parameter(nonneg=True, value = trade, name="trade_cost") # value=0.005 is .5%
        self.risk = cp.Parameter(nonneg=True, value = risk, name='risk_aversion')

        # Variables

        self.u = cp.Variable((T, N+1))
        self.x = [cp.Variable((K, N+1), nonneg = True) for _ in range(T)]  # scenario number, time, ith asset - and no shorting

        # AUX variables
        self.z = cp.Variable() # expected final wealth
        
        # negative and positive trades
        self.pos, self.neg = cp.Variable((T, N+1), nonneg=True), cp.Variable((T, N+1), nonneg=True) 

        # Constraints
        
        constraints = [self.x[0] == self.x0]
        #constraints += [x[:, T-1, 1:] <= 1e-4] # End with cash only

        constraints.append(self.pos - self.neg == self.u)

        for t in range(T-1):

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


        final_wealth = cp.sum(self.x[T-1], axis=1)  # shape (K,)

        constraints.append(self.z == cp.sum(final_wealth) / K)
        
        if risk_type == 'variance':
            risk_measure = cp.var(self.z - final_wealth)

        if risk_type == 'semi_variance':
            self.downside = cp.Variable(K, nonneg=True)
            constraints += [self.downside >= self.z - final_wealth]
            risk_measure = cp.var(self.downside)

        objective = cp.Maximize(
            self.z - self.risk * risk_measure
            )

        self.problem = cp.Problem(objective, constraints)

    def is_QP_and_DPP(self):
        return self.problem.is_qp(), self.problem.is_dpp()
    
    def update_parameters(self, scenarios, x0):
        for t, r_t in enumerate(self.r):
            r_t.value = scenarios[:, t, :]
        
        self.x0.value = x0
            

    def solve(self):
        self.problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            polish=True,
            eps_abs=1e-2,
            eps_rel=1e-2,
            rho=0.1,
            max_iter=2000
        )

        if self.problem.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            raise RuntimeError(f"Problem is {self.problem.status}")
        
        return {
            'objective': self.problem.value,
            'status': self.problem.status,
            'solve_time': self.problem.solver_stats.solve_time
        }
    
    def control(self):
        """Return first control action"""
        return self.u[0].value.copy()
        
    
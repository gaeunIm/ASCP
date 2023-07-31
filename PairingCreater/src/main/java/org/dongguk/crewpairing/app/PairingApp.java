package org.dongguk.crewpairing.app;

import org.dongguk.crewpairing.domain.*;
import org.dongguk.crewpairing.persistence.FlightCrewPairingGenerator;
import org.dongguk.crewpairing.util.ViewAllConstraint;
import org.optaplanner.core.api.score.ScoreExplanation;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.constraint.ConstraintMatchTotal;
import org.optaplanner.core.api.solver.SolutionManager;
import org.optaplanner.core.api.solver.Solver;
import org.optaplanner.core.api.solver.SolverFactory;

import java.util.*;

public class PairingApp {
    public static String SOLVER_CONFIG = "solverConfig.xml";
    public static void main(String[] args) {
        // 최대 pairing 수 정하기
        Scanner scanner = new Scanner(System.in);
        System.out.print("Plz Input PairingSetSize: ");
        int pairingSetSize = Integer.parseInt(scanner.nextLine());

        // Solver 생성
        SolverFactory<PairingSolution> solverFactory = SolverFactory.createFromXmlResource("solverConfig.xml");
        FlightCrewPairingGenerator generator = new FlightCrewPairingGenerator();

        // Load the problem
        PairingSolution problem = generator.createInput(pairingSetSize);

        // Solve the problem
        Solver<PairingSolution> solver = solverFactory.buildSolver();
        PairingSolution solution = solver.solve(problem);

        // Visualize the solution
        System.out.println(solution);
        
        // OutPut Excel
        generator.createOutput(solution);
//        PairingVisualize.visualize(solution.getPairingList());

        // Check score detail
        SolutionManager<PairingSolution, HardSoftScore> scoreManager = SolutionManager.create(solverFactory);
        ScoreExplanation<PairingSolution, HardSoftScore> explain = scoreManager.explain(solution);
        Map<String, ConstraintMatchTotal<HardSoftScore>> constraintMatchTotalMap = explain.getConstraintMatchTotalMap();
        ViewAllConstraint.viewAll(constraintMatchTotalMap, solution);

        System.exit(0);
    }
}

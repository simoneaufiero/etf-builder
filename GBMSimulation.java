/**
 * 
 */
package com.lorenzoberti.session03;

import net.finmath.exception.CalculationException;
import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.RandomVariableFromDoubleArray;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

public class GBMSimulation {
	
	public static void main(String[] args) throws CalculationException {

		int numberOfPaths = 1;
		int numberOfFactors = 4;
		int seed = 14320;

		double initialTime = 0.0;
		double finalTime = 100.0;
		double deltaT = 1.0;
		int numberOfTimeSteps = (int) (finalTime / deltaT);

		double firstAssetInitial = 10.0;
		double secondAssetInitial = 12.0;
		double thirdAssetInitial = 14.0;
		double fourthAssetInitial = 16.0;

		double firstAssetVolDouble = 0.3;
		double secondAssetVolDouble = 0.5;
		double thirdAssetVolDouble = 0.1;
		double fourthAssetVolDouble = 0.2;

		double riskFree = 0.05;
		
		double correlationFactor12 = 0.2;
		double correlationFactor13 = 0.6;
		double correlationFactor14 = 0.4;
		//double correlationFactor23 = 0.001;
		//double correlationFactor24 = -0.8;
		//double correlationFactor34 = -0.3;
		
		double choleskyElement1 = 0.9798;
		double choleskyElement2 = -0.3804;
		double choleskyElement3 = 0.6714;
		double choleskyElement4 = -0.8185;
		double choleskyElement5 = 0.2980;
		double choleskyElement6 = 0.8455;

		RandomVariable firstAssetInitialValue = new RandomVariableFromDoubleArray(firstAssetInitial);
		RandomVariable secondAssetInitialValue = new RandomVariableFromDoubleArray(secondAssetInitial);
		RandomVariable thirdAssetInitialValue = new RandomVariableFromDoubleArray(thirdAssetInitial);
		RandomVariable fourthAssetInitialValue = new RandomVariableFromDoubleArray(fourthAssetInitial);

		RandomVariable firstAssetVol = new RandomVariableFromDoubleArray(firstAssetVolDouble);
		RandomVariable secondAssetVol = new RandomVariableFromDoubleArray(secondAssetVolDouble);
		RandomVariable thirdAssetVol = new RandomVariableFromDoubleArray(thirdAssetVolDouble);
		RandomVariable fourthAssetVol = new RandomVariableFromDoubleArray(fourthAssetVolDouble);

		TimeDiscretization times = new TimeDiscretizationFromArray(initialTime, numberOfTimeSteps, deltaT);

		BrownianMotion brownian = new BrownianMotionFromMersenneRandomNumbers(times, numberOfFactors, numberOfPaths, seed);
		
		DoubleTernaryOperator geometricBrownian = (x, y, z) -> {
			return Math.exp((riskFree - y * y * 0.5) * z + y * x);
		};
		
		RandomVariable[] firstBrownian = new RandomVariable[100];
		RandomVariable[] secondBrownian = new RandomVariable[100];
		RandomVariable[] thirdBrownian = new RandomVariable[100];
		RandomVariable[] fourthBrownian = new RandomVariable[100];
		
		RandomVariable[] firstAsset = new RandomVariable[100];
		RandomVariable[] secondAsset = new RandomVariable[100];
		RandomVariable[] thirdAsset = new RandomVariable[100];
		RandomVariable[] fourthAsset = new RandomVariable[100];
		
		firstBrownian[0]= new RandomVariableFromDoubleArray(0);
		secondBrownian[0]= new RandomVariableFromDoubleArray(0);
		thirdBrownian[0]= new RandomVariableFromDoubleArray(0);
		fourthBrownian[0]= new RandomVariableFromDoubleArray(0);
		
		firstAsset[0] = firstAssetInitialValue;
		secondAsset[0] = secondAssetInitialValue;
		thirdAsset[0] = thirdAssetInitialValue;
		fourthAsset[0] = fourthAssetInitialValue;
		
		
		for (int i = 1; i < 100; i++) {
			firstBrownian[i] = firstBrownian[i - 1].add(brownian.getBrownianIncrement(i, 0));
			
			RandomVariable actualTime = new RandomVariableFromDoubleArray(i);
						
			firstAsset[i] = firstAssetInitialValue.mult(firstBrownian[i].apply(geometricBrownian, firstAssetVol, actualTime));
        }
		
		for (int i = 1; i < 100; i++) {
			secondBrownian[i] = secondBrownian[i - 1].add(firstBrownian[i].mult(correlationFactor12)
					.add(brownian.getBrownianIncrement(i, 1).mult(choleskyElement1)));
			
			RandomVariable actualTime = new RandomVariableFromDoubleArray(i);
			
			secondAsset[i] = secondAssetInitialValue.mult(secondBrownian[i].apply(geometricBrownian, secondAssetVol, actualTime));
        }
		
		for (int i = 1; i < 100; i++) {
			thirdBrownian[i] = thirdBrownian[i - 1].add(firstBrownian[i].mult(correlationFactor13)
					.add(brownian.getBrownianIncrement(i, 1).mult(choleskyElement2))
					.add(brownian.getBrownianIncrement(i, 2).mult(choleskyElement3)));
			
			RandomVariable actualTime = new RandomVariableFromDoubleArray(i);
			
			thirdAsset[i] = thirdAssetInitialValue.mult(thirdBrownian[i].apply(geometricBrownian, thirdAssetVol, actualTime));
        }
		
		for (int i = 1; i < 100; i++) {
			fourthBrownian[i] = fourthBrownian[i - 1].add(firstBrownian[i].mult(correlationFactor14)
					.add(brownian.getBrownianIncrement(i, 1).mult(choleskyElement4))
					.add(brownian.getBrownianIncrement(i, 2).mult(choleskyElement5))
					.add(brownian.getBrownianIncrement(i, 3).mult(choleskyElement6)));
			
			RandomVariable actualTime = new RandomVariableFromDoubleArray(i);
			
			fourthAsset[i] = fourthAssetInitialValue.mult(fourthBrownian[i].apply(geometricBrownian, fourthAssetVol, actualTime));
        }
 		
		for (int i = 0; i < 100; i++) {
		System.out.println("1" + firstBrownian[i]);
		}
		
		for (int i = 0; i < 100; i++) {
		System.out.println("2" + secondBrownian[i]);
		}
		
		for (int i = 0; i < 100; i++) {
		System.out.println("3" + thirdAsset[i]);
		}
		
		for (int i = 0; i < 100; i++) {
		System.out.println("4" + fourthAsset[i]);
		}
		
}
}

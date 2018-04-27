from sklearn import neighbors #our ai
from random import randint
import pandas as pd #helps with data wrangling
from gplearn.genetic import SymbolicTransformer #needed for genetic data science
from sklearn.model_selection import train_test_split, StratifiedKFold #helps with data wrangling
from sklearn.model_selection import cross_val_score  # needed for testing the ai
import numpy as np #helps with data wrangling
#randomState = (randint(50, 100))
randomState = 33 #needed for debugging

#ACQUIRE AND CLEAN DATA
rawData = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data") #get raw census data
cleanedData = rawData[rawData.columns[5:]]#drop state, county, community, and city dimensions. they hold no signal
for column in cleanedData.columns: #missing data is a legitimate problem you must work around
    if True in cleanedData[column].isin(["?"]).tolist(): #the missing data is labeled with string "?" in this particular data set
        cleanedData.drop(column, axis=1, inplace=True) #i will simply drop the dimensions with missing values

#BUILD DATA
dimensions = cleanedData[cleanedData.columns[:len(cleanedData.columns) - 1]] #create a universe with hundreds of dimensions...economics, family structure, race, gender, etc
def makeTargets(args):
    if  args < cleanedData[cleanedData.columns[-1]].mean(): #if the city's crime rate is below the average, we define this target as low crime, or 0
        return 0
    if  args > cleanedData[cleanedData.columns[-1]].mean():#if the city's crime rate is above the average, we define this target as high crime, or 1
        return 1
targets = cleanedData[cleanedData.columns[-1]].apply(makeTargets) #this is what we want to predict. our target is last column in the data set "ViolentCrimesPerPop"
examSize = .25 #the size of our test for the ai
studyGuideQuestions, examQuestions, studyGuideAnswers, examAnswers = train_test_split( #split the data into 2 sets, training(learning) and testing(confirming the signal is real)
    dimensions.as_matrix(), targets.as_matrix(), #pass in our dimensions(predictors like economics, race, gender). also pass in our targets (violent crime)
    test_size=examSize, shuffle=False, random_state=randomState) #keep shuffling and random state the same so that we can debug our results without the dataset order changing

#MACHINE LEARNING
ai = neighbors.KNeighborsClassifier(n_neighbors=9) #this is the ai that we send into hyperspace data so it can learn the signal, then we'll use it to predict future crime rates
ai.fit(studyGuideQuestions, studyGuideAnswers) #make the ai find the signal ("studying" for the exam)
print "Baseline testing score: " + str(ai.score(examQuestions, examAnswers)) #now we make the ai take the "exam". is its accuracy just as good as when it was trained? yes! it found true signal
cv = StratifiedKFold(random_state=randomState, n_splits=5, shuffle=False) #beyond the scope of this tutorial
baselineTrainingScore = cross_val_score(ai, studyGuideQuestions, studyGuideAnswers, cv=cv).mean()#we will use this later to compare the fitness of genetic dimensions on our full training set

#EVOLVE DATA
evolution = SymbolicTransformer( #our habitat for breeding species(formulas) that predict our target (violent crime)
                                generations=20, #how many generations of competition/sex we want our species to live through
                                population_size=20000, #each generation will have this many species(formulas) that try to out-evolve each other
                                function_set=['mul', 'div', 'min', 'max', 'add', 'sub'], #the weapons(math operators) our species use to compete against one another.
                                init_depth=(5,7), #the complexity of each species, or how many "weapons" from the function_set each species gets initially
                                init_method="full", #similar to line above
                                tournament_size=2, #competition. how many species should compare(fight) fitness against each other at once. strength building lifeforce
                                p_crossover=0.9, #sex. when the winners breed, this is the probability of "inventing" a new hybrid species. adaptive lifeforce
                                parsimony_coefficient="auto", #controls how complex each species can become, so that they can still breed with each other and have successful crossovers(mutations)
                                hall_of_fame=1000, #after all generations have happened, select the top ranking species. this what we're looking for
                                n_components=5, #of the top ranking species, take even a smaller subset. programmatically i dont understand the difference between this parameter and the one above
                                verbose=1, #coding parameter. make the program tell us info about its evolution as it runs
                                random_state=randomState, #coding parameter. keep random seed state the same so we can debug
                                )
evolution.fit(studyGuideQuestions, studyGuideAnswers) #run the evolution. again, this generates random dimension formulas(species) and picks the best ones that help predict our target(violent crime)
geneticDimensions = evolution.transform(dimensions) #now we have the genetic dimensions. next we'll rerun the ai using the top genetic diminsions plus the original dimensions then take the final exam
finalDimensions = dimensions #coding stuff. placeholder var
for alpha in range(geneticDimensions.shape[1]): #for each "alpha" species dimension generated by our evolution
    tempDimensions =  np.hstack((geneticDimensions[:, [alpha]], dimensions)) #add it to the original dimensions (gender, race, economics)
    studyGuideQuestions, examQuestions, studyGuideAnswers, examAnswers = train_test_split(tempDimensions, targets.as_matrix(), test_size=examSize, shuffle=False, random_state=randomState)
    #ai.fit(studyGuideQuestions, studyGuideAnswers) #make the ai study again using the new alpha genetic dimension
    geneticDimensionScore = cross_val_score(ai, studyGuideQuestions, studyGuideAnswers, cv=cv).mean() #make the ai study with this new alpha dimension
    if geneticDimensionScore > baselineTrainingScore: #did the ai do better using the new genetic dimension as opposed to just the normal dimensions?
        finalDimensions =  np.hstack((geneticDimensions[:, [alpha]], finalDimensions)) # ..if so, then we will make the ai use this alpha genetic dimension later when it takes the "final exam" (test set)
        print "Alpha dimension: " + str (evolution._best_programs[alpha]) #print the actual alpha species(formula) itself because this shit is cool af

studyGuideQuestions, examQuestions, studyGuideAnswers, examAnswers = train_test_split( #make new training and testing sets using the old dimensions and the full set of new genetic alpha dimensions
    finalDimensions, targets.as_matrix(), test_size=examSize, shuffle=False, random_state=randomState)
ai.fit(studyGuideQuestions, studyGuideAnswers) #make the ai study for its exam, but this time we hopefully gave it better studying techniques...regular dimensions(race,gender,etc) AND genetic dimensions
print "Genetic dimensions test score: " + str(ai.score(examQuestions, examAnswers)) #make the ai take a final exam. did it score better than when it just used normal dimensions? yes!!!!!!

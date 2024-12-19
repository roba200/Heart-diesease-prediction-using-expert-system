from experta import *

# Define the expert system knowledge base
class HeartDiseaseExpertSystem(KnowledgeEngine):
    @Rule(Fact(cp=P(lambda x: x <= 0.5)) & Fact(ca=P(lambda x: x <= 0.5)) & Fact(thal=P(lambda x: x <= 2.5)))
    def rule_1(self):
        self.declare(Fact(disease=1))
        print("Rule 1: Likely to have heart disease.")

    @Rule(Fact(cp=P(lambda x: x <= 0.5)) & Fact(ca=P(lambda x: x <= 0.5)) & Fact(thal=P(lambda x: x > 2.5)))
    def rule_2(self):
        self.declare(Fact(disease=0))
        print("Rule 2: Unlikely to have heart disease.")

    @Rule(Fact(cp=P(lambda x: x <= 0.5)) & Fact(ca=P(lambda x: x > 0.5)) & Fact(trestbps=P(lambda x: x <= 109)))
    def rule_3(self):
        self.declare(Fact(disease=0))
        print("Rule 3: Unlikely to have heart disease.")

    @Rule(Fact(cp=P(lambda x: x <= 0.5)) & Fact(ca=P(lambda x: x > 0.5)) & Fact(trestbps=P(lambda x: x > 109)))
    def rule_4(self):
        self.declare(Fact(disease=0))
        print("Rule 4: Unlikely to have heart disease.")

    @Rule(Fact(cp=P(lambda x: x > 0.5)) & Fact(thal=P(lambda x: x <= 2.5)) & Fact(oldpeak=P(lambda x: x <= 2.1)))
    def rule_5(self):
        self.declare(Fact(disease=1))
        print("Rule 5: Likely to have heart disease.")

    @Rule(Fact(cp=P(lambda x: x > 0.5)) & Fact(thal=P(lambda x: x <= 2.5)) & Fact(oldpeak=P(lambda x: x > 2.1)))
    def rule_6(self):
        self.declare(Fact(disease=0))
        print("Rule 6: Unlikely to have heart disease.")

    @Rule(Fact(cp=P(lambda x: x > 0.5)) & Fact(thal=P(lambda x: x > 2.5)) & Fact(thalach=P(lambda x: x <= 132.5)))
    def rule_7(self):
        self.declare(Fact(disease=0))
        print("Rule 7: Unlikely to have heart disease.")

    @Rule(Fact(cp=P(lambda x: x > 0.5)) & Fact(thal=P(lambda x: x > 2.5)) & Fact(thalach=P(lambda x: x > 132.5)))
    def rule_8(self):
        self.declare(Fact(disease=1))
        print("Rule 8: Likely to have heart disease.")

# Function to get inputs from the user
def get_user_inputs():
    print("Enter the following values:")
    cp = float(input("Chest Pain Type (0-3): "))
    ca = float(input("Number of Major Vessels (0-3): "))
    thal = float(input("Thalassemia (0-3): "))
    trestbps = float(input("Resting Blood Pressure (mm Hg): "))
    oldpeak = float(input("ST Depression: "))
    thalach = float(input("Max Heart Rate Achieved: "))
    return cp, ca, thal, trestbps, oldpeak, thalach

# Instantiate and run the expert system
engine = HeartDiseaseExpertSystem()
engine.reset()

# Get user inputs
cp, ca, thal, trestbps, oldpeak, thalach = get_user_inputs()

# Declare facts based on user inputs
engine.declare(Fact(cp=cp), Fact(ca=ca), Fact(thal=thal), Fact(trestbps=trestbps), Fact(oldpeak=oldpeak), Fact(thalach=thalach))

# Run the expert system
engine.run()
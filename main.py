from experta import *

# Define the Knowledge Base
class HeartDiseaseExpertSystem(KnowledgeEngine):
    @DefFacts()
    def initial_facts(self):
        yield Fact(action="determine_risk")
        print("Expert System Initialized for Heart Disease Prediction.\n")

    @Rule(Fact(action="determine_risk"), NOT(Fact(age=W())))
    def ask_age(self):
        self.declare(Fact(age=int(input("Enter your age: "))))

    @Rule(Fact(action="determine_risk"), NOT(Fact(gender=W())))
    def ask_gender(self):
        self.declare(Fact(gender=input("Enter your gender (male/female): ").lower()))

    @Rule(Fact(action="determine_risk"), NOT(Fact(chest_pain=W())))
    def ask_chest_pain(self):
        self.declare(Fact(chest_pain=input("Do you experience chest pain (yes/no)? ").lower()))

    @Rule(Fact(action="determine_risk"), NOT(Fact(blood_pressure=W())))
    def ask_blood_pressure(self):
        self.declare(Fact(blood_pressure=int(input("Enter your blood pressure level: "))))

    @Rule(Fact(action="determine_risk"), NOT(Fact(cholesterol=W())))
    def ask_cholesterol(self):
        self.declare(Fact(cholesterol=int(input("Enter your cholesterol level: "))))

    @Rule(Fact(action="determine_risk"), 
          Fact(age=MATCH.age), 
          Fact(gender=MATCH.gender), 
          Fact(chest_pain="yes"), 
          Fact(blood_pressure=MATCH.bp & P(lambda bp: bp > 140)),
          Fact(cholesterol=MATCH.chol & P(lambda chol: chol > 200)))
    def high_risk(self, age, gender, bp, chol):
        print(f"\nBased on the inputs, the patient ({age} years old, {gender}) is at HIGH RISK of heart disease.")
        self.declare(Fact(risk="high"))

    @Rule(Fact(action="determine_risk"), 
          Fact(age=MATCH.age), 
          Fact(gender=MATCH.gender), 
          Fact(chest_pain="no"), 
          Fact(blood_pressure=MATCH.bp & P(lambda bp: bp <= 140)),
          Fact(cholesterol=MATCH.chol & P(lambda chol: chol <= 200)))
    def low_risk(self, age, gender, bp, chol):
        print(f"\nBased on the inputs, the patient ({age} years old, {gender}) is at LOW RISK of heart disease.")
        self.declare(Fact(risk="low"))

    @Rule(Fact(action="determine_risk"), Fact(risk=MATCH.risk))
    def conclusion(self, risk):
        if risk == "high":
            print("Consult a cardiologist immediately for further evaluation.")
        elif risk == "low":
            print("Maintain a healthy lifestyle and consider regular check-ups.")

# Initialize and run the expert system
if __name__ == "__main__":
    engine = HeartDiseaseExpertSystem()
    engine.reset()  # Prepare the engine for execution
    engine.run()    # Start the engine
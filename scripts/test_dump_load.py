#!/usr/bin/env python3

from aha import Model, Trainer

model_json = """
{
  "r": 2,
  "d": 2,
  "w": [0.4, 0.6],
  "c": [
    {
      "u": [1.0, 2.0],
      "s": [1.0, 0.0, 0.0, 1.0]
    },
    {
      "u": [0.5, -0.5],
      "s": [2.0, 0.5, 0.5, 1.25]
    }
  ]
}
"""


def test_model():
    model = Model()
    print(model.Import(model_json))
    print(model.Load(model.Dump()))
    print(model.Export())
    
    
trainer_json = """
{
  "r": 2,
  "d": 2,
  "e": -500.0,
  "w": [40.0, 60.0],
  "m": [
    [1.0, 1.0],
    [2.0, 2.0]
  ],
  "c": [
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0]
  ]
}
"""

def test_trainer():
    model = Model(rank=2, dim=2)
    trainer = Trainer(model)
    trainer.Reset()
    print(trainer.Swallow(trainer_json))
    print(trainer.Load(trainer.Dump()))
    print(trainer.Spit())


if __name__ == "__main__":
    test_model()
    test_trainer()
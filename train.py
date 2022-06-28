from trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.test()
    trainer.save_model()

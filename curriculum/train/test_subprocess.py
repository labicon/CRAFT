import subprocess

process = subprocess.run(["python", 
                            "/home/kang/MANavigation/skrl_ws/curriculum_train.py", 
                            "--run_date", "05-05_17-35",
                            "--curriculum_task", "1_Basic Navigation(05-05_17-35)", 
                            "--sample_idx", "0",
                            "--training_iter", "5000",
                            ])
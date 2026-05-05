import time

print("Hello! I am a Python script running inside a Docker Container.")
print("I don't need any special web frameworks to run.")

for i in range(5):
    print(f"Working... step {i+1}/5")
    time.sleep(1)

print("Job finished! The container will now stop because my process is complete.")

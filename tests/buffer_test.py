from src.replay_buffer import ReplayBuffer


buffer = ReplayBuffer(capacity=100)

# Add some fake experiences
for i in range(10):
    state = [i, i+1, i+2, i+3]  # Fake state
    action = i % 2              # Fake action (0 or 1)
    reward = 1.0                # Fake reward
    next_state = [i+1, i+2, i+3, i+4]  # Fake next state
    done = False
    
    buffer.add(state, action, reward, next_state, done)

print(f"Buffer size: {len(buffer)}")  # Should be 10
batch = buffer.sample(3)               # Sample 3 experiences
print(f"Batch size: {len(batch)}")    # Should be 3
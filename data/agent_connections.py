from agentnode.core.agent import Agent

# Create and connect agent
agent = Agent(name="MyAgent")
await agent.connect_to_server("127.0.0.1", 8765)

# Send messages to other agents
await agent.send_message("other_agent_id", "Hello!")

## Updated at:
2026-05-18T05:00:15.815-04:00

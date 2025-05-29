"""
The purpose of this script is to demonstrate how to use the new OOP interface for Synapse AI Agents.

1. Register and send a prompt to a custom agent
2. Send a prompt to the baseline Synapse Agent
3. Conduct more than one session with the same agent
4. Start a new session with a custom agent and send a prompt to it
5. Start a new session with the baseline Synapse Agent and send a prompt to it
6. Start a new session with a custom agent and then update what the agent has access to
"""

import os
import sys
import synapseclient
from synapseclient.models import Agent, AgentSession, AgentSessionAccessLevel

# IDs for a bedrock agent with the instructions:
# "You are a test agent that when greeted with: 'hello' will always response with: 'world'"
CLOUD_AGENT_ID = "QOTV3KQM1X"
AGENT_REGISTRATION_ID = 29

SYNAPSE_TOKEN = os.getenv("SYNAPSE_AUTH_TOKEN")  # Synapse personal access token (or OAuth access token)

# Simple validation of credentials
if not SYNAPSE_TOKEN:
    print("ERROR: Synapse auth token not provided. Set the SYNAPSE_AUTH_TOKEN environment variable.", file=sys.stderr)
    sys.exit(1)
# if not FH_API_KEY:
#     print("ERROR: FutureHouse API key not provided. Set the FH_API_KEY environment variable.", file=sys.stderr)
#     sys.exit(1)

def authenticate_synapse(token: str) -> synapseclient.Synapse:
    """
    Authenticate to Synapse using a PAT/OAuth token and return a Synapse client object.
    """
    try:
        syn = synapseclient.Synapse()  # Create Synapse client instance
        # Login with the provided token (using authToken avoids needing username/password):contentReference[oaicite:33]{index=33}
        syn.login(authToken=token, silent=True)
        # The 'silent=True' prevents printing the welcome message to stdout.
        return syn
    except Exception as e:
        # If authentication fails, raise an error with details
        raise RuntimeError(f"Synapse authentication failed: {e}")

#syn = synapseclient.Synapse(debug=True)
# Authenticate to Synapse
syn = authenticate_synapse(SYNAPSE_TOKEN)
#syn.login()

# Using the Agent class

# Start a new session with the baseline Synapse Agent and send a prompt to it
def start_new_session_with_baseline_agent_and_send_prompt_to_it():
    my_session = AgentSession().start(synapse_client=syn)
    my_session.access_level = AgentSessionAccessLevel.WRITE_YOUR_PRIVATE_DATA
    my_session.update(synapse_client=syn)
    my_session.prompt(
        prompt="Download the file with ID syn26401298. Then save into the directory /Users/jineta/git/gitrepo/data-analysis-crow ",
        enable_trace=True,
        print_response=True,
        synapse_client=syn,
    )


start_new_session_with_baseline_agent_and_send_prompt_to_it()
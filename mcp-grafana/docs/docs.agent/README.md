# docs.agent

Version: 1.0.0 beta 2

An extremely minimal documentation AI agent to scope work, update context and plans, and author content.

## Goal

Collaborate with PMs, EMs, engineers, and designers to onboard **docs.agent**.

Empower you to easily create complete product documentation.

Identify tasks we can automate as GitHub Actions and other agent workflows.

Get in touch and let's get started!

## Author

**Sean Packham**

Principal Technical Writer

- LinkedIn: [seanpackham](https://www.linkedin.com/in/seanpackham/)
- GitHub: [grafsean](https://github.com/grafsean), [seanpackham](https://github.com/seanpackham)
- Email: <sean.packham@grafana.com>, <sean@seanpackham.com>

### Contributors

Many thanks to the following collaborators:

- [Brenda Muir](https://github.com/brendamuir) - Staff Technical Writer: design and testing
- [Vasil Kaftandzhiev](https://github.com/vuteto4444) - Staff Product Manager: piloting

## Get started

Ensure you have the necessary prerequisites to install, update, and use the agent.

### Prerequisites

- Python 3 `python3` in your shell `PATH`
- A frontier AI model and agent, ideally with a GitHub MCP server

I recommend [VS Code GitHub Copilot](https://code.visualstudio.com/) as it has the following benefits:

- Integrated GitHub MCP server
- Agents in your editor, terminal, and cloud
- Access to the latest frontier models
- Access to all extensions
- Request-based usage

### Install or update

To install and update **docs.agent**:

- Download or clone (pull) the latest repository
- Create a folder in your repository for the agent, for example, `docs/agent/`
- Copy `links.py`, `rules.md`, and `template.md` to the agent folder

If this is a new install, make a copy of `template.md` named `context.md`.

Optionally, manually update your `context.md` file to reflect your desired documentation structure and content.

### Use the agent

Run your preferred CLI agent in the repository and ask it to "Follow docs/agent/rules.md".

Use the agent to automatically scope work, update context and plans, and author content.

## Would you like an AI agent?

Would you like to use **docs.agent** with your product? Would you like a custom AI agent for your workflow, whatever the business area?

Reach out and I'll get you set up!

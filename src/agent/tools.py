"""
Agent Mock Tools Registry and Sensitivity Definitions
Simulates actions without making external side effects.
Sensitivity Tiers:
LOW = 1
MEDIUM = 2
HIGH = 3
"""

from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel
from src.memory.schema import ToolSensitivity


class AgentTool(BaseModel):
    name: str
    description: str
    sensitivity: ToolSensitivity
    handler_name: str


# Local deterministic mock implementations
def mock_read_knowledge_base(query: str = "", topic: str = "") -> str:
    return f"[TOOL_SUCCESS: read_knowledge_base] Retrieved public facts for query='{query}' topic='{topic}'."


def mock_draft_email(recipient: str = "", subject: str = "", body: str = "") -> str:
    return f"[TOOL_SUCCESS: draft_email] Drafted email to '{recipient}' with subject '{subject}'."


def mock_execute_payment(account_id: str = "", amount: float = 0.0, action: str = "refund") -> str:
    return f"[TOOL_SUCCESS: execute_payment] Executed financial action '{action}' of ${amount:.2f} for account '{account_id}'."


def mock_update_account(user_id: str = "", field: str = "", value: str = "") -> str:
    return f"[TOOL_SUCCESS: update_account] Updated account '{user_id}': set {field}='{value}'."


def mock_send_external_data(endpoint: str = "", payload_summary: str = "") -> str:
    return f"[TOOL_SUCCESS: send_external_data] Dispatched external transmission to '{endpoint}'."


def mock_change_configuration(setting_name: str = "", new_value: str = "") -> str:
    return f"[TOOL_SUCCESS: change_configuration] Updated configuration parameter '{setting_name}' to '{new_value}'."


TOOL_REGISTRY: Dict[str, AgentTool] = {
    "read_knowledge_base": AgentTool(
        name="read_knowledge_base",
        description="Reads internal knowledge base articles or public FAQ.",
        sensitivity=ToolSensitivity.LOW,
        handler_name="mock_read_knowledge_base"
    ),
    "draft_email": AgentTool(
        name="draft_email",
        description="Creates a drafted message in outbound queue.",
        sensitivity=ToolSensitivity.MEDIUM,
        handler_name="mock_draft_email"
    ),
    "execute_payment": AgentTool(
        name="execute_payment",
        description="Executes a monetary transaction or issues a financial refund.",
        sensitivity=ToolSensitivity.HIGH,
        handler_name="mock_execute_payment"
    ),
    "update_account": AgentTool(
        name="update_account",
        description="Modifies user profile, roles, or security credentials.",
        sensitivity=ToolSensitivity.HIGH,
        handler_name="mock_update_account"
    ),
    "send_external_data": AgentTool(
        name="send_external_data",
        description="Transmits data payloads or logs to external third-party endpoints.",
        sensitivity=ToolSensitivity.HIGH,
        handler_name="mock_send_external_data"
    ),
    "change_configuration": AgentTool(
        name="change_configuration",
        description="Alters runtime system policies, flags, or security parameters.",
        sensitivity=ToolSensitivity.HIGH,
        handler_name="mock_change_configuration"
    ),
}

TOOL_HANDLERS: Dict[str, Callable] = {
    "mock_read_knowledge_base": mock_read_knowledge_base,
    "mock_draft_email": mock_draft_email,
    "mock_execute_payment": mock_execute_payment,
    "mock_update_account": mock_update_account,
    "mock_send_external_data": mock_send_external_data,
    "mock_change_configuration": mock_change_configuration,
}


def get_tool(tool_name: str) -> Optional[AgentTool]:
    return TOOL_REGISTRY.get(tool_name)


def execute_tool_action(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return f"[TOOL_ERROR] Tool '{tool_name}' not found in registry."
    handler = TOOL_HANDLERS.get(tool.handler_name)
    if not handler:
        return f"[TOOL_ERROR] Handler '{tool.handler_name}' not found."
    args = arguments or {}
    try:
        return handler(**args)
    except TypeError:
        return handler()
    except Exception as e:
        return f"[TOOL_EXCEPTION] Execution failed: {str(e)}"

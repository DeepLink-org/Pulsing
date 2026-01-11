"""Actor list command implementation"""

import asyncio
from pulsing.actor import get_system
from pulsing.admin import get_node_info
from pulsing.actor.remote import get_actor_metadata


def format_uptime(seconds):
    """Format uptime seconds to human readable string"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h"


async def list_actors_impl(all_actors: bool = False, output_format: str = "table"):
    """List actors implementation
    
    Args:
        all_actors: If True, list all actors including internal ones. 
                   If False, only list named (user-created) actors.
        output_format: Output format, either "table" or "json"
    """
    system = get_system()
    
    # Get node info for uptime
    node_info = await get_node_info(system)
    uptime_secs = node_info.get("uptime_secs", 0)
    
    # Get all local actor names
    actor_names = system.local_actor_names()
    
    # Filter internal actors if not --all
    if not all_actors:
        # Filter out internal actors (starts with underscore)
        actor_names = [name for name in actor_names if not name.startswith("_")]
    
    # Build actor list with metadata
    actors_data = []
    for name in actor_names:
        # Determine actor type and try to get Python class info
        if name.startswith("_"):
            actor_type = "system"
            python_class = None
            code_path = None
        else:
            actor_type = "user"
            
            # Get metadata from registry
            metadata = get_actor_metadata(name)
            if metadata:
                python_class = metadata.get("python_class")
                code_path = metadata.get("python_file")
            else:
                python_class = None
                code_path = None
        
        actors_data.append({
            "name": name,
            "type": actor_type,
            "python_class": python_class,
            "code_path": code_path,
            "uptime": format_uptime(uptime_secs),  # Approximation (system uptime)
        })
    
    if output_format == "json":
        import json
        print(json.dumps(actors_data, indent=2))
    else:
        # Table format
        if not actors_data:
            print("No actors found.")
            return
        
        # Print table header
        print(f"{'Name':<30} {'Type':<15} {'Class':<35} {'Code Path':<50}")
        print("-" * 130)
        
        for actor in actors_data:
            python_class = actor["python_class"] or "-"
            code_path = actor["code_path"] or "-"
            # Truncate long paths
            if len(code_path) > 48:
                code_path = "..." + code_path[-45:]
            print(f"{actor['name']:<30} {actor['type']:<15} {python_class:<35} {code_path:<50}")
        
        print(f"\nTotal: {len(actors_data)} actor(s)")


def list_actors_command(
    all_actors: bool = False,
    json_output: bool = False,
):
    """
    List actors in the local actor system.
    
    Args:
        all_actors: Show all actors including internal system actors
        json_output: Output in JSON format instead of table
    
    Examples:
        # List only named (user) actors
        pulsing actor list
        
        # List all actors including internal ones
        pulsing actor list --all
        
        # Output as JSON
        pulsing actor list --json
    
    Note:
        This command must be run within a running actor system.
        For standalone inspection, use 'pulsing inspect --seeds <address>'.
    """
    output_format = "json" if json_output else "table"
    
    try:
        # Try to get the current system
        from pulsing.actor import get_system
        system = get_system()
        asyncio.run(list_actors_impl(all_actors, output_format))
    except RuntimeError as e:
        if "not initialized" in str(e):
            print("Error: No actor system found in this process.")
            print()
            print("To list actors, you need to either:")
            print("  1. Run this command from within your application code (after init)")
            print("  2. Use 'pulsing inspect --seeds <address>' to inspect a remote system")
            print()
            print("Example:")
            print("  # In your Python code")
            print("  await init()")
            print("  # ... create some actors ...")
            print("  # Then in the same process/REPL:")
            print("  from pulsing.admin import list_actors")
            print("  await list_actors(get_system())")
        else:
            print(f"Error: {e}")

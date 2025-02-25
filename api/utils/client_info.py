from fastapi import Request
import re
from typing import Optional
from api.schemas.request import ClientInfo

# Regular expressions for detecting mobile, tablet and desktop devices
MOBILE_PATTERN = re.compile(r"Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini", re.IGNORECASE)
TABLET_PATTERN = re.compile(r"iPad|Android(?!.*Mobile)|Tablet", re.IGNORECASE)

# Regular expressions for detecting browsers
BROWSER_PATTERNS = {
    "Chrome": re.compile(r"Chrome/([\d\.]+)"),
    "Firefox": re.compile(r"Firefox/([\d\.]+)"),
    "Safari": re.compile(r"Safari/([\d\.]+)"),
    "Edge": re.compile(r"Edge/([\d\.]+)"),
    "Opera": re.compile(r"Opera/([\d\.]+)|OPR/([\d\.]+)"),
    "IE": re.compile(r"MSIE ([\d\.]+)|Trident.*rv:([\d\.]+)"),
    "Samsung": re.compile(r"SamsungBrowser/([\d\.]+)"),
    "UC": re.compile(r"UCBrowser/([\d\.]+)"),
}

# Regular expressions for detecting operating systems
OS_PATTERNS = {
    "Windows": re.compile(r"Windows NT ([\d\.]+)"),
    "macOS": re.compile(r"Mac OS X ([\d_\.]+)"),
    "iOS": re.compile(r"iPhone OS ([\d_]+)"),
    "Android": re.compile(r"Android ([\d\.]+)"),
    "Linux": re.compile(r"Linux"),
}


def extract_client_info(request: Request) -> Optional[ClientInfo]:
    """
    Extract client device and browser information from the request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ClientInfo object or None if user agent is not available
    """
    # Get user agent string
    user_agent = request.headers.get("user-agent", "")
    
    if not user_agent:
        return None
        
    # Determine device type
    device_type = "desktop"  # Default
    
    if TABLET_PATTERN.search(user_agent):
        device_type = "tablet"
    elif MOBILE_PATTERN.search(user_agent):
        device_type = "mobile"
        
    # Determine browser
    browser_name = "Unknown"
    browser_version = None
    
    for name, pattern in BROWSER_PATTERNS.items():
        match = pattern.search(user_agent)
        if match:
            browser_name = name
            # Get the first non-None group
            browser_version = next((g for g in match.groups() if g is not None), None)
            break
            
    # Determine OS
    os_name = "Unknown"
    os_version = None
    
    for name, pattern in OS_PATTERNS.items():
        match = pattern.search(user_agent)
        if match:
            os_name = name
            # Format version for iOS and macOS (replace underscores with dots)
            version = next((g for g in match.groups() if g is not None), None)
            if version and ("_" in version):
                version = version.replace("_", ".")
            os_version = version
            break
            
    # Get screen dimensions from headers or set to None
    screen_width = None
    screen_height = None
    
    # Try to get language
    language = request.headers.get("accept-language", "").split(",")[0] if request.headers.get("accept-language") else None
    
    # Create ClientInfo object
    return ClientInfo(
        device_type=device_type,
        os_name=os_name,
        os_version=os_version,
        browser_name=browser_name,
        browser_version=browser_version,
        screen_width=screen_width,
        screen_height=screen_height,
        user_agent=user_agent,
        ip_address=request.client.host if request.client else None,
        language=language
    )


def get_device_category(client_info: ClientInfo) -> str:
    """
    Get a simple device category from client info.
    
    Args:
        client_info: ClientInfo object
        
    Returns:
        str: 'mobile', 'tablet', or 'desktop'
    """
    if not client_info:
        return "unknown"
        
    return client_info.device_type or "unknown"


def get_browser_family(client_info: ClientInfo) -> str:
    """
    Get browser family from client info.
    
    Args:
        client_info: ClientInfo object
        
    Returns:
        str: Browser family name
    """
    if not client_info or not client_info.browser_name:
        return "unknown"
        
    return client_info.browser_name


def get_os_family(client_info: ClientInfo) -> str:
    """
    Get OS family from client info.
    
    Args:
        client_info: ClientInfo object
        
    Returns:
        str: OS family name
    """
    if not client_info or not client_info.os_name:
        return "unknown"
        
    return client_info.os_name
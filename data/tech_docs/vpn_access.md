# VPN Access and Troubleshooting

## What is VPN?

Secure remote access to:
- File servers and shared drives
- Company databases and applications
- Development/staging environments
- Intranet and internal websites

## Requirements

- Windows 10/11, macOS 11+, or approved Linux
- Company-issued laptop (personal devices not supported)
- Cisco AnyConnect client
- Stable internet (minimum 5 Mbps, no public WiFi)

## Connecting to VPN

### First-Time Setup
1. Open Cisco AnyConnect (pre-installed or download from https://vpn.company.com)
2. Server address: `vpn.company.com` → Click "Connect"
3. Enter company email, password, and MFA code

### Daily Use
1. Launch Cisco AnyConnect → Click "Connect"
2. Enter credentials and MFA code
3. Wait for "Connected" status (5-10 seconds)

### Disconnecting
- Click "Disconnect" or auto-disconnects after 8 hours of inactivity

## Multi-Factor Authentication (MFA)

### Methods
- **Microsoft Authenticator** (recommended): Push notification or time-based code
- **SMS Code**: 6-digit code to registered mobile
- **Hardware Token**: Available for frequent travelers

### Setup
- Go to https://mfa.company.com
- Register at least 2 methods for backup

## Troubleshooting

### "Connection Failed" or "Connection Timeout"
1. Check internet connection and VPN status: https://status.company.com
2. Try different internet (mobile hotspot)
3. Temporarily disable personal firewall/antivirus
4. Contact IT if persists

### "Invalid Credentials"
- Verify username is complete email address
- Ensure Caps Lock is off
- Verify MFA code is current (expires every 30 seconds)
- Clear saved credentials and re-enter

### "Certificate Error"
- Ensure system date/time is correct
- Update Cisco AnyConnect
- Contact IT to refresh certificate

### VPN Connects But Can't Access Resources
- Verify full connection (green indicator)
- Disconnect and reconnect
- Restart computer

### Slow Performance
- Test internet speed (need 5+ Mbps)
- Try off-peak hours
- Close unnecessary applications
- Use wired connection if possible

## Split Tunneling

**Work traffic through VPN:**
- Company email/calendar
- Internal websites and applications
- File servers and databases

**Direct internet access:**
- Web browsing (non-internal sites)
- Personal email, streaming, social media

## Security Best Practices

✓ Connect to VPN for all company resources
✓ Disconnect when done
✓ Use secure networks only (home, mobile hotspot)
✓ Keep VPN client updated

✗ Never share credentials
✗ No public WiFi
✗ No third-party VPN software

## Support

**Available**: 24/7 | **Maintenance**: Sundays 2-4 AM EST, First Saturday monthly 12-6 AM EST

- **Status**: https://status.company.com
- **Phone**: ext. 5555 or 1-800-555-0199
- **Email**: helpdesk@company.com
- **Emergency**: 1-800-555-0911

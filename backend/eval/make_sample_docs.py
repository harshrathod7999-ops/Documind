"""Generate a realistic PDF corpus for the DocuMind eval.

Three fictional documents, sized and cross-linked so retrieval is a *real*
task (~20+ chunks with many near-duplicate distractors) rather than trivial:

  * atlas_x200_manual.pdf      — the product the eval questions are about
  * atlas_x100_manual.pdf      — an OLDER sibling with DIFFERENT specs/codes →
                                 strong semantic distractors; exact model/code
                                 matching (BM25) must win over dense similarity
  * acmenet_support_policy.pdf — support/returns/warranty (conflicts on warranty)

The X100 deliberately mirrors the X200's structure with different numbers, so
dense-only search tends to confuse the two; BM25 + cross-encoder rerank pull the
correct section back. That contrast is what makes the before/after table real.

Run:  python eval/make_sample_docs.py   ->   eval/sample_docs/*.pdf
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).parent / "sample_docs"

# --- Document 1: Atlas X200 (subject of the eval questions) ---------------
X200_MANUAL: list[tuple[str, str]] = [
    ("Overview",
     "The Atlas X200 is an industrial-grade router designed for harsh "
     "environments such as factory floors, outdoor cabinets, and transport "
     "depots. It operates between -20C and 65C and supports both 2.4GHz and "
     "5GHz wireless bands. The default management IP address is 192.168.10.1 "
     "and the default admin username is 'atlas-admin'. The device ships with "
     "firmware version 4.8.2. The X200 is the successor to the older Atlas "
     "X100 and is not backward compatible with X100 power supplies or mounting "
     "brackets. A migration guide is available for customers moving from the "
     "X100 platform to preserve their existing configuration profiles."),
    ("Specifications",
     "The Atlas X200 provides 8 Gigabit Ethernet ports and 2 SFP+ ports rated "
     "at 10 Gbps. Maximum throughput is 9.4 Gbps. Power draw is 38 watts "
     "typical and 55 watts peak. The unit weighs 1.6 kilograms and uses a 12V "
     "DC input. Wireless range is up to 120 meters outdoors. The chassis is "
     "rated IP40 for dust and is fanless below 40 watts of load. Onboard "
     "storage is 8 GB of eMMC for logs and configuration snapshots. The "
     "operating humidity range is 5 to 95 percent non-condensing."),
    ("Installation",
     "Mount the Atlas X200 on a DIN rail or use the included wall bracket. "
     "Allow at least 4 centimeters of clearance on each side for airflow. "
     "Connect the WAN cable to port 1, which is the only port that supports "
     "Power over Ethernet input. The first boot takes approximately 90 seconds "
     "while the device initializes its configuration database. After the first "
     "boot, log in at the default management IP and immediately change the "
     "admin password. Mounting the unit upside down is not supported and will "
     "impair passive cooling."),
    ("Networking",
     "The Atlas X200 supports VLAN tagging per the 802.1Q standard, up to 256 "
     "VLANs. It includes a stateful firewall, NAT, and an IPsec VPN that "
     "supports up to 50 concurrent tunnels. Dynamic routing uses OSPF and BGP. "
     "The DHCP server can hand out up to 4000 leases. DNS caching is enabled "
     "by default and can be disabled from the Advanced Networking menu. Quality "
     "of Service supports eight priority queues mapped to DSCP markings."),
    ("Wireless Configuration",
     "The Atlas X200 radio can run in access-point or bridge mode. In "
     "access-point mode it broadcasts up to four SSIDs per band, each mappable "
     "to a separate VLAN. WPA3 is the default security mode; WPA2 is available "
     "for legacy clients. The recommended channel width is 80 MHz on the 5GHz "
     "band. Transmit power is adjustable from 1 to 23 dBm. Band steering nudges "
     "dual-band clients toward the 5GHz band when the signal is adequate."),
    ("Error Codes",
     "Error code E-4021 indicates a thermal shutdown caused by overheating; "
     "the unit must be powered off and allowed to cool for 10 minutes. Error "
     "code E-3300 indicates a failed firmware checksum and requires reflashing "
     "the firmware. Error code E-1002 means the WAN link is down. Error code "
     "E-5500 indicates a fan failure and the device should be serviced "
     "immediately. Error code E-2200 indicates that the configuration database "
     "is corrupt and a factory reset is recommended. Error code E-6010 "
     "indicates an expired security certificate on the management interface."),
    ("LED Indicators",
     "A solid green status LED means normal operation. A blinking amber LED "
     "means the device is booting or applying a configuration. A solid red LED "
     "indicates a hardware fault that maps to an error code in the system log. "
     "The WAN LED is off when no link is detected and blinks to show traffic. "
     "The two SFP+ ports each have a dedicated link LED that is green at 10 "
     "Gbps and amber when negotiating a lower speed."),
    ("Maintenance",
     "Inspect the cooling fans every 90 days. Replace the air filter, part "
     "number SKU-9981, every 180 days in dusty environments. Firmware should "
     "be updated quarterly. To factory reset, hold the recessed reset button "
     "for 15 seconds until the status LED blinks amber. Back up the "
     "configuration before any firmware update. The real-time clock battery is "
     "a CR2032 and should be replaced every five years to preserve log "
     "timestamps across power loss."),
    ("Troubleshooting",
     "If the device will not power on, verify the 12V DC supply and check that "
     "the barrel connector is fully seated. If wireless clients cannot connect, "
     "confirm the correct band is enabled and that the channel is not "
     "congested. If throughput is below expectations, disable DNS caching and "
     "verify that hardware offload is enabled under Advanced Networking. If the "
     "management interface is unreachable, connect over the console port at "
     "115200 baud and inspect the boot log for error codes."),
    ("Security",
     "Administrative access can be restricted by source IP and protected with "
     "TOTP-based two-factor authentication. The Atlas X200 logs all "
     "configuration changes with the operator username and a timestamp. "
     "Firmware images are signed; an invalid signature raises error code "
     "E-3300 and the image is rejected. Disable the legacy HTTP interface and "
     "use HTTPS only for production deployments."),
    ("Error Code Reference",
     "The following is the complete Atlas X200 error code reference. "
     "E-1002 means the WAN link is down. E-1003 means a LAN port has been "
     "disabled by loop protection. E-1101 means a DHCP address pool is "
     "exhausted. E-1450 means an IPsec tunnel failed to negotiate. E-2200 "
     "means the configuration database is corrupt. E-2201 means a configuration "
     "restore failed due to a version mismatch. E-3300 means a firmware "
     "checksum failed. E-3301 means a firmware downgrade was blocked. E-4021 "
     "means a thermal shutdown from overheating. E-4022 means the intake "
     "temperature sensor is disconnected. E-5500 means a fan failure. E-5501 "
     "means a fan is spinning below its rated speed. E-6010 means a security "
     "certificate on the management interface has expired. E-6011 means the "
     "certificate chain is incomplete. E-7000 means an unsupported SFP module "
     "was inserted."),
    ("Frequently Asked Questions",
     "How do I reset a forgotten admin password? Hold the reset button for 15 "
     "seconds to factory reset, which restores the default admin-admin "
     "credentials. Can the X200 be powered over Ethernet? Yes, but only on "
     "port 1, and only as a PoE input. Does the X200 support stacking? No, the "
     "X200 does not support hardware stacking; use dynamic routing instead. "
     "What is the maximum number of VPN tunnels? Up to 50 concurrent IPsec "
     "tunnels. How many DHCP leases can it serve? Up to 4000 leases. Is the "
     "X100 air filter compatible? No, the X200 uses filter SKU-9981 while the "
     "X100 uses SKU-4410."),
    ("Compliance and Power",
     "The Atlas X200 is certified to FCC Part 15 Class A and CE EN 55032. The "
     "external power adapter is rated 12V DC at 5 amps. Inrush current at power "
     "on can reach 8 amps for under 10 milliseconds. The unit supports a soft "
     "shutdown via the management interface that flushes logs to eMMC before "
     "powering down. Mean time between failures is rated at 200,000 hours at "
     "25C ambient."),
    ("Warranty",
     "The Atlas X200 hardware is covered by a limited warranty of 36 months "
     "from the date of purchase. The warranty does not cover damage from power "
     "surges, liquid ingress, or unauthorized firmware modification. Warranty "
     "claims require the original proof of purchase and the device serial "
     "number printed on the base plate. Replacement units are shipped within "
     "five business days of an approved claim."),
]

# --- Document 2: Atlas X100 (DISTRACTOR — older, different numbers) --------
X100_MANUAL: list[tuple[str, str]] = [
    ("Overview",
     "The Atlas X100 is the previous-generation industrial router. It operates "
     "between -10C and 55C and supports only the 2.4GHz band. The default "
     "management IP address is 192.168.1.1 and the default admin username is "
     "'admin'. The device ships with firmware version 3.2.0. The X100 has been "
     "superseded by the Atlas X200 and is no longer sold, though security "
     "patches are issued through the end of its support window."),
    ("Specifications",
     "The Atlas X100 provides 4 Fast Ethernet ports and no SFP ports. Maximum "
     "throughput is 940 Mbps. Power draw is 18 watts typical. The unit weighs "
     "0.9 kilograms and uses a 9V DC input. Wireless range is up to 60 meters "
     "outdoors. The chassis is rated IP30 and is passively cooled with no fans. "
     "Onboard storage is 2 GB."),
    ("Networking",
     "The Atlas X100 supports up to 16 VLANs and a basic stateless firewall. "
     "It does not support BGP; only static routes and RIP are available. The "
     "DHCP server can hand out up to 500 leases. There is no hardware offload, "
     "so throughput drops under heavy firewall rule counts."),
    ("Error Codes",
     "On the Atlas X100, error code E-4021 does not exist. Error code E-101 "
     "indicates a power fault, error code E-202 indicates a wireless radio "
     "failure, and error code E-303 indicates the WAN link is down on the "
     "X100. These codes are not compatible with the X200 error code scheme, "
     "and cross-referencing them will produce incorrect repair actions."),
    ("Maintenance",
     "Inspect the Atlas X100 enclosure every 120 days. The X100 air filter is "
     "part number SKU-4410 and is not interchangeable with the X200 filter. "
     "There are no user-serviceable fans in the X100 because it is passively "
     "cooled. Hold the reset button for 8 seconds to factory reset the X100. "
     "Firmware updates for the X100 are delivered twice a year."),
    ("Warranty",
     "The Atlas X100 hardware warranty is 24 months from the date of purchase. "
     "Extended warranty options are no longer sold for the discontinued X100, "
     "and out-of-warranty repairs are handled on a best-effort basis only."),
]

# --- Document 3: AcmeNet support policy (conflicts on warranty length) -----
SUPPORT_POLICY: list[tuple[str, str]] = [
    ("Support Tiers",
     "AcmeNet offers three support tiers: Standard, Priority, and Mission "
     "Critical. Standard support responds within 2 business days. Priority "
     "support responds within 4 hours during business hours. Mission Critical "
     "support responds within 1 hour and includes 24/7 phone access and a "
     "named technical account manager."),
    ("Return Policy",
     "Hardware may be returned within 30 days of delivery for a full refund. "
     "After 30 days, returns are subject to a 15 percent restocking fee. "
     "Opened software licenses are non-refundable. Return shipping is paid by "
     "the customer unless the unit was dead on arrival, in which case AcmeNet "
     "provides a prepaid label."),
    ("Extended Warranty",
     "Under the AcmeNet extended care plan, the Atlas X200 warranty is extended "
     "to 60 months. Note this differs from the standard 36 month hardware "
     "warranty stated in the product manual; the longer term applies only when "
     "the extended care plan is purchased separately within 90 days of the "
     "original hardware purchase. The extended plan also includes advance "
     "hardware replacement."),
    ("Escalation",
     "Unresolved Priority tickets are automatically escalated to a senior "
     "engineer after 8 hours. Mission Critical incidents page the on-call "
     "engineering lead immediately and open a war-room bridge with a dedicated "
     "incident commander. Customers can request a post-incident review for any "
     "Mission Critical outage."),
    ("Service Credits",
     "If AcmeNet misses a Mission Critical response target, the customer "
     "receives a service credit equal to 10 percent of the monthly support "
     "fee per missed incident, capped at 50 percent in any single month. "
     "Service credits must be claimed within 30 days of the missed target."),
    ("Data Handling",
     "Support engineers may request diagnostic bundles that include "
     "configuration and logs. Diagnostic data is retained for 90 days and then "
     "deleted. Customers can request immediate deletion in writing. AcmeNet "
     "does not access customer traffic payloads during support sessions."),
]


def _write_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    w = pdf.epw
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(w, 10, title)
    pdf.ln(4)
    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(w, 8, heading)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(w, 6, body)
        pdf.ln(3)
    pdf.output(str(path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_pdf(OUT / "atlas_x200_manual.pdf", "Atlas X200 Product Manual", X200_MANUAL)
    _write_pdf(OUT / "atlas_x100_manual.pdf", "Atlas X100 Product Manual", X100_MANUAL)
    _write_pdf(OUT / "acmenet_support_policy.pdf", "AcmeNet Support Policy", SUPPORT_POLICY)
    print(f"Wrote 3 sample PDFs to {OUT}")


if __name__ == "__main__":
    main()

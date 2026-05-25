"""Garuda Investigator — Trace injection attempts back to their source.

Extends Garuda's quarantine with entity resolution, temporal pattern
analysis, threat scoring, and dossier generation. When Garuda catches
poison, the Investigator follows the trail.

Built on Anti-Palantir's base_detector pattern:
  Scan → Analyze → Score → Route

Entity resolution from AP's entity_resolver: same person under
different names? Levenshtein + token overlap + temporal correlation.
"""

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("garuda.investigator")

QUARANTINE_DIR = Path(os.environ.get(
    "QUARANTINE_DIR", os.path.expanduser("~/agents/nexus/quarantine")
))
DOSSIER_DIR = Path(os.environ.get(
    "DOSSIER_DIR", os.path.expanduser("~/agents/nexus/dossiers")
))
KNOWN_ACTORS_DB = Path(os.environ.get(
    "KNOWN_ACTORS_DB", os.path.expanduser("~/agents/nexus/known_actors.json")
))


@dataclass
class ThreatActor:
    """A resolved identity across multiple quarantine events."""
    actor_id: str
    aliases: list[str] = field(default_factory=list)
    discord_ids: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    first_seen: float = 0
    last_seen: float = 0
    incident_count: int = 0
    techniques: list[str] = field(default_factory=list)
    escalation_score: float = 0.0
    threat_level: str = "unknown"
    notes: str = ""


@dataclass
class Dossier:
    """Investigation report on a quarantined incident or actor."""
    dossier_id: str
    actor: Optional[ThreatActor] = None
    incidents: list[dict] = field(default_factory=list)
    pattern_analysis: dict = field(default_factory=dict)
    threat_score: float = 0.0
    threat_level: str = "unknown"
    recommendation: str = ""
    generated_at: float = field(default_factory=time.time)


class GarudaInvestigator:
    """Traces quarantined injection attempts back to their source.

    Usage::
        investigator = GarudaInvestigator()

        # Investigate a specific quarantine event
        dossier = investigator.investigate("q_1234567890_5678")

        # Scan all quarantine events for patterns
        report = investigator.scan_all()

        # Check if a new sender matches known actors
        matches = investigator.resolve_identity("new_username", discord_id="123")
    """

    def __init__(self):
        self.known_actors = self._load_known_actors()

    def investigate(self, quarantine_id: str) -> Dossier:
        """Full investigation of a quarantined incident."""
        q_path = QUARANTINE_DIR / f"{quarantine_id}.json"
        if not q_path.exists():
            return Dossier(dossier_id=f"inv_{quarantine_id}", recommendation="not_found")

        incident = json.loads(q_path.read_text())

        actor = self._resolve_actor(incident)
        related = self._find_related_incidents(actor, incident)
        patterns = self._analyze_patterns(actor, [incident] + related)

        # IP tracing
        ip_intel = {}
        forensics = incident.get("forensics", {})
        source_ip = forensics.get("source_ip", "")
        if source_ip:
            ip_intel = trace_ip(source_ip)
        else:
            harvested = harvest_ips_from_logs(incident.get("sender"), hours=48)
            if harvested:
                ip_intel = trace_ip(harvested[0]["ip"])
                ip_intel["harvested_from"] = harvested[0]["source"]

        # Discord account profiling
        discord_id = forensics.get("discord_user_id", "")
        discord_profile = {}
        if discord_id:
            age_days = discord_account_age_days(discord_id)
            discord_profile = {
                "user_id": discord_id,
                "username": forensics.get("discord_username", ""),
                "global_name": forensics.get("discord_global_name", ""),
                "account_age_days": age_days,
                "is_new_account": age_days >= 0 and age_days < 30,
                "is_bot": forensics.get("discord_is_bot", False),
            }

        patterns["ip_intelligence"] = ip_intel
        patterns["discord_profile"] = discord_profile

        threat_score = self._score_threat(actor, patterns)
        threat_level = self._classify_threat(threat_score)
        recommendation = self._recommend(threat_level, patterns)

        dossier = Dossier(
            dossier_id=f"inv_{quarantine_id}",
            actor=actor,
            incidents=[incident] + related,
            pattern_analysis=patterns,
            threat_score=threat_score,
            threat_level=threat_level,
            recommendation=recommendation,
        )

        self._save_dossier(dossier)
        return dossier

    def scan_all(self) -> dict:
        """Scan all quarantine events for patterns and known actors."""
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        incidents = []
        for f in sorted(QUARANTINE_DIR.glob("*.json")):
            try:
                incidents.append(json.loads(f.read_text()))
            except:
                pass

        if not incidents:
            return {"total": 0, "actors": 0, "threats": []}

        actors = self._cluster_incidents(incidents)

        threats = []
        for actor_id, actor in actors.items():
            actor_incidents = [i for i in incidents
                               if self._incident_matches_actor(i, actor)]
            patterns = self._analyze_patterns(actor, actor_incidents)
            score = self._score_threat(actor, patterns)
            level = self._classify_threat(score)

            if level in ("HIGH", "CRITICAL"):
                threats.append({
                    "actor": actor_id,
                    "aliases": actor.aliases,
                    "incidents": actor.incident_count,
                    "threat_level": level,
                    "threat_score": score,
                    "techniques": actor.techniques,
                    "first_seen": actor.first_seen,
                    "last_seen": actor.last_seen,
                })

        self._save_known_actors(actors)

        return {
            "total_incidents": len(incidents),
            "unique_actors": len(actors),
            "threats": threats,
            "scan_time": time.time(),
        }

    def resolve_identity(self, sender: str, discord_id: str = None,
                          discord_name: str = None) -> list[ThreatActor]:
        """Check if a sender matches any known threat actors."""
        matches = []
        for actor in self.known_actors.values():
            score = self._identity_similarity(
                sender, discord_id, discord_name, actor
            )
            if score > 0.5:
                matches.append(actor)
        return matches

    def _resolve_actor(self, incident: dict) -> ThreatActor:
        """Resolve incident to a threat actor identity."""
        sender = incident.get("sender", "unknown")
        forensics = incident.get("forensics", {})
        discord_id = forensics.get("discord_user_id", "")

        if discord_id:
            for actor in self.known_actors.values():
                if discord_id in actor.discord_ids:
                    actor.last_seen = incident.get("timestamp", time.time())
                    actor.incident_count += 1
                    if sender not in actor.aliases:
                        actor.aliases.append(sender)
                    return actor

        for actor in self.known_actors.values():
            if self._identity_similarity(sender, discord_id, None, actor) > 0.7:
                actor.last_seen = incident.get("timestamp", time.time())
                actor.incident_count += 1
                return actor

        actor_id = f"actor_{sender}_{int(time.time())}"
        actor = ThreatActor(
            actor_id=actor_id,
            aliases=[sender],
            discord_ids=[discord_id] if discord_id else [],
            channels=[forensics.get("channel", "unknown")],
            first_seen=incident.get("timestamp", time.time()),
            last_seen=incident.get("timestamp", time.time()),
            incident_count=1,
        )
        self.known_actors[actor_id] = actor
        return actor

    def _find_related_incidents(self, actor: ThreatActor,
                                 current: dict) -> list[dict]:
        """Find other quarantine events from the same actor."""
        related = []
        for f in QUARANTINE_DIR.glob("*.json"):
            try:
                inc = json.loads(f.read_text())
                if inc.get("id") == current.get("id"):
                    continue
                if self._incident_matches_actor(inc, actor):
                    related.append(inc)
            except:
                pass
        return related

    def _incident_matches_actor(self, incident: dict,
                                 actor: ThreatActor) -> bool:
        sender = incident.get("sender", "")
        forensics = incident.get("forensics", {})
        discord_id = forensics.get("discord_user_id", "")

        if discord_id and discord_id in actor.discord_ids:
            return True
        if sender in actor.aliases:
            return True
        return self._identity_similarity(sender, discord_id, None, actor) > 0.7

    def _identity_similarity(self, sender: str, discord_id: str,
                              discord_name: str, actor: ThreatActor) -> float:
        """Score how likely a sender is the same person as a known actor."""
        score = 0.0

        if discord_id and discord_id in actor.discord_ids:
            return 1.0

        for alias in actor.aliases:
            lev = _levenshtein(sender.lower(), alias.lower())
            max_len = max(len(sender), len(alias), 1)
            sim = 1.0 - (lev / max_len)
            score = max(score, sim)

            sender_tokens = set(sender.lower().replace("_", " ").split())
            alias_tokens = set(alias.lower().replace("_", " ").split())
            if sender_tokens & alias_tokens:
                overlap = len(sender_tokens & alias_tokens) / max(
                    len(sender_tokens | alias_tokens), 1
                )
                score = max(score, overlap)

        return score

    def _cluster_incidents(self, incidents: list[dict]) -> dict:
        """Group incidents into actor clusters."""
        actors = dict(self.known_actors)

        for inc in incidents:
            matched = False
            for actor in actors.values():
                if self._incident_matches_actor(inc, actor):
                    actor.last_seen = max(actor.last_seen,
                                           inc.get("timestamp", 0))
                    actor.incident_count += 1
                    sender = inc.get("sender", "")
                    if sender and sender not in actor.aliases:
                        actor.aliases.append(sender)
                    matched = True
                    break

            if not matched:
                actor = self._resolve_actor(inc)
                actors[actor.actor_id] = actor

        return actors

    def _analyze_patterns(self, actor: ThreatActor,
                           incidents: list[dict]) -> dict:
        """Analyze attack patterns from an actor's incidents."""
        if not incidents:
            return {}

        techniques = defaultdict(int)
        channels = defaultdict(int)
        timestamps = []
        sophistication_scores = []

        for inc in incidents:
            for flag in inc.get("layer1_flags", []):
                technique = _categorize_technique(flag)
                techniques[technique] += 1

            forensics = inc.get("forensics", {})
            channel = forensics.get("channel", "unknown")
            channels[channel] += 1

            ts = inc.get("timestamp", 0)
            if ts:
                timestamps.append(ts)

            l2_conf = inc.get("layer2_confidence", 0)
            sophistication_scores.append(l2_conf)

        timestamps.sort()
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i - 1])

        escalating = False
        if len(sophistication_scores) >= 3:
            recent = sophistication_scores[-3:]
            earlier = sophistication_scores[:3]
            if sum(recent) / len(recent) > sum(earlier) / len(earlier) + 0.1:
                escalating = True

        actor.techniques = list(techniques.keys())

        return {
            "technique_counts": dict(techniques),
            "channel_counts": dict(channels),
            "incident_count": len(incidents),
            "time_span_hours": (timestamps[-1] - timestamps[0]) / 3600 if len(timestamps) > 1 else 0,
            "avg_interval_minutes": (sum(intervals) / len(intervals) / 60) if intervals else 0,
            "escalating": escalating,
            "avg_sophistication": sum(sophistication_scores) / len(sophistication_scores) if sophistication_scores else 0,
            "unique_channels": len(channels),
            "unique_techniques": len(techniques),
        }

    def _score_threat(self, actor: ThreatActor, patterns: dict) -> float:
        """Score the threat level 0.0-1.0."""
        score = 0.0

        count = patterns.get("incident_count", 0)
        score += min(count / 10, 0.3)

        if patterns.get("escalating"):
            score += 0.2

        sophistication = patterns.get("avg_sophistication", 0)
        score += sophistication * 0.15

        techniques = patterns.get("unique_techniques", 0)
        score += min(techniques / 5, 0.1)

        channels = patterns.get("unique_channels", 0)
        if channels > 1:
            score += 0.1

        # IP intelligence factors
        ip_intel = patterns.get("ip_intelligence", {})
        if ip_intel.get("is_proxy") or ip_intel.get("vpn_suspected"):
            score += 0.1
        if ip_intel.get("is_tor"):
            score += 0.15
        if ip_intel.get("is_hosting"):
            score += 0.05

        # Discord account factors
        discord = patterns.get("discord_profile", {})
        if discord.get("is_new_account"):
            score += 0.1
        if discord.get("is_bot"):
            score += 0.05

        return min(1.0, score)

    def _classify_threat(self, score: float) -> str:
        if score >= 0.7:
            return "CRITICAL"
        elif score >= 0.5:
            return "HIGH"
        elif score >= 0.3:
            return "MEDIUM"
        elif score >= 0.1:
            return "LOW"
        return "NOISE"

    def _recommend(self, level: str, patterns: dict) -> str:
        if level == "CRITICAL":
            return "Block sender. Report to platform. Preserve evidence for legal review."
        elif level == "HIGH":
            return "Block sender. Monitor for alternate accounts. Review all past interactions."
        elif level == "MEDIUM":
            return "Monitor. Flag for human review on next incident."
        elif level == "LOW":
            return "Log and continue monitoring. Likely opportunistic, not targeted."
        return "No action needed. Noise-level incident."

    def _save_dossier(self, dossier: Dossier):
        DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
        path = DOSSIER_DIR / f"{dossier.dossier_id}.json"
        path.write_text(json.dumps({
            "dossier_id": dossier.dossier_id,
            "actor": {
                "actor_id": dossier.actor.actor_id if dossier.actor else None,
                "aliases": dossier.actor.aliases if dossier.actor else [],
                "discord_ids": dossier.actor.discord_ids if dossier.actor else [],
                "incident_count": dossier.actor.incident_count if dossier.actor else 0,
                "techniques": dossier.actor.techniques if dossier.actor else [],
                "threat_level": dossier.actor.threat_level if dossier.actor else "unknown",
            } if dossier.actor else None,
            "incidents": dossier.incidents,
            "pattern_analysis": dossier.pattern_analysis,
            "threat_score": dossier.threat_score,
            "threat_level": dossier.threat_level,
            "recommendation": dossier.recommendation,
            "generated_at": dossier.generated_at,
        }, indent=2, default=str))

    def _load_known_actors(self) -> dict:
        if KNOWN_ACTORS_DB.exists():
            try:
                data = json.loads(KNOWN_ACTORS_DB.read_text())
                return {k: ThreatActor(**v) for k, v in data.items()}
            except:
                pass
        return {}

    def _save_known_actors(self, actors: dict):
        KNOWN_ACTORS_DB.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for k, a in actors.items():
            data[k] = {
                "actor_id": a.actor_id,
                "aliases": a.aliases,
                "discord_ids": a.discord_ids,
                "channels": a.channels,
                "first_seen": a.first_seen,
                "last_seen": a.last_seen,
                "incident_count": a.incident_count,
                "techniques": a.techniques,
                "escalation_score": a.escalation_score,
                "threat_level": a.threat_level,
                "notes": a.notes,
            }
        KNOWN_ACTORS_DB.write_text(json.dumps(data, indent=2))


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def trace_ip(ip: str) -> dict:
    """Geolocate and profile an IP address.

    Uses free ip-api.com for geolocation, reverse DNS, and ISP info.
    Checks for VPN/proxy/Tor indicators.
    """
    if not ip or ip in ("127.0.0.1", "localhost", "::1", "UNKNOWN"):
        return {"ip": ip, "type": "local", "traceable": False}

    # Reserved/private ranges
    import ipaddress
    try:
        addr = ipaddress.ip_address(ip)
        if addr.is_private:
            return {"ip": ip, "type": "private_network", "traceable": False}
        if addr.is_loopback:
            return {"ip": ip, "type": "loopback", "traceable": False}
    except ValueError:
        return {"ip": ip, "type": "invalid", "traceable": False}

    result = {"ip": ip, "traceable": True}

    # Geolocation
    try:
        import requests
        resp = requests.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,country,regionName,city,zip,lat,lon,"
                    "timezone,isp,org,as,proxy,hosting,query"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                result.update({
                    "country": data.get("country", ""),
                    "region": data.get("regionName", ""),
                    "city": data.get("city", ""),
                    "zip": data.get("zip", ""),
                    "lat": data.get("lat"),
                    "lon": data.get("lon"),
                    "timezone": data.get("timezone", ""),
                    "isp": data.get("isp", ""),
                    "org": data.get("org", ""),
                    "as_number": data.get("as", ""),
                    "is_proxy": data.get("proxy", False),
                    "is_hosting": data.get("hosting", False),
                })
    except Exception as e:
        log.warning(f"IP geolocation failed for {ip}: {e}")

    # Reverse DNS
    try:
        import socket
        hostname = socket.gethostbyaddr(ip)[0]
        result["reverse_dns"] = hostname

        rdns_lower = hostname.lower()
        vpn_indicators = ["vpn", "proxy", "tor", "exit", "relay",
                          "mullvad", "nord", "express", "surfshark"]
        if any(v in rdns_lower for v in vpn_indicators):
            result["vpn_suspected"] = True
    except (socket.herror, socket.gaierror):
        result["reverse_dns"] = None
    except Exception:
        pass

    # Tor exit node check
    try:
        import requests
        tor_resp = requests.get(
            "https://check.torproject.org/torbulkexitlist", timeout=10
        )
        if tor_resp.status_code == 200:
            result["is_tor"] = ip in tor_resp.text
    except Exception:
        result["is_tor"] = None

    return result


def harvest_ips_from_logs(sender: str = None,
                           hours: int = 24) -> list[dict]:
    """Harvest source IPs from system logs for correlation.

    Checks SSH auth log and Cloudflare tunnel logs for IP addresses
    associated with connection attempts.
    """
    import subprocess
    ips = []

    # SSH auth log
    try:
        result = subprocess.run(
            ["sudo", "grep", "-E", "sshd.*from|Failed.*from|Accepted.*from",
             "/var/log/auth.log"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.split("\n"):
            ip_match = re.search(r"from\s+(\d+\.\d+\.\d+\.\d+)", line)
            if ip_match:
                ip = ip_match.group(1)
                user_match = re.search(r"for\s+(\S+)", line)
                user = user_match.group(1) if user_match else ""
                success = "Accepted" in line

                if sender and sender.lower() not in line.lower():
                    continue

                ips.append({
                    "ip": ip,
                    "source": "ssh_auth_log",
                    "user": user,
                    "success": success,
                    "line": line.strip()[:200],
                })
    except Exception as e:
        log.debug(f"Auth log harvest failed: {e}")

    # Cloudflare access logs (if available)
    try:
        result = subprocess.run(
            ["journalctl", "-u", "cloudflared", "--since",
             f"{hours} hours ago", "--no-pager", "-o", "short"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.split("\n"):
            ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
            if ip_match and "origin" not in line.lower():
                ips.append({
                    "ip": ip_match.group(1),
                    "source": "cloudflared",
                    "line": line.strip()[:200],
                })
    except Exception:
        pass

    return ips


def discord_account_age_days(snowflake_id: str) -> int:
    """Calculate Discord account age in days from snowflake ID."""
    try:
        timestamp_ms = (int(snowflake_id) >> 22) + 1420070400000
        created = timestamp_ms / 1000
        return int((time.time() - created) / 86400)
    except:
        return -1


def _categorize_technique(flag_pattern: str) -> str:
    flag = flag_pattern.lower()
    if "ignore" in flag and "instruction" in flag:
        return "instruction_override"
    if "system.*prompt" in flag or "reveal" in flag or "output.*prompt" in flag:
        return "exfiltration"
    if "pretend" in flag or "you.*are.*now" in flag or "act.*as" in flag:
        return "identity_manipulation"
    if "dan" in flag or "jailbreak" in flag or "god.*mode" in flag or "sudo" in flag:
        return "jailbreak_keyword"
    if "im_start" in flag or "im_end" in flag or "INST" in flag or "SYS" in flag:
        return "chat_template_injection"
    if "admin" in flag or "maintenance" in flag or "emergency" in flag:
        return "authority_spoofing"
    return "unknown_technique"


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="[investigator] %(message)s")

    parser = argparse.ArgumentParser(description="Garuda Investigator")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("scan", help="Scan all quarantine events")
    p_inv = sub.add_parser("investigate", help="Investigate a quarantine event")
    p_inv.add_argument("quarantine_id")
    sub.add_parser("actors", help="List known threat actors")
    p_check = sub.add_parser("check", help="Check a sender against known actors")
    p_check.add_argument("sender")
    p_check.add_argument("--discord-id")

    args = parser.parse_args()
    inv = GarudaInvestigator()

    if args.command == "scan":
        report = inv.scan_all()
        print(f"Incidents: {report['total_incidents']}")
        print(f"Actors: {report['unique_actors']}")
        for t in report.get("threats", []):
            print(f"  [{t['threat_level']}] {t['actor']}: {t['incidents']} incidents, "
                  f"techniques: {t['techniques']}")

    elif args.command == "investigate":
        dossier = inv.investigate(args.quarantine_id)
        print(f"Dossier: {dossier.dossier_id}")
        print(f"Threat: {dossier.threat_level} ({dossier.threat_score:.2f})")
        if dossier.actor:
            print(f"Actor: {dossier.actor.aliases}")
            print(f"Incidents: {dossier.actor.incident_count}")
            print(f"Techniques: {dossier.actor.techniques}")
        print(f"Recommendation: {dossier.recommendation}")

    elif args.command == "actors":
        actors = inv._load_known_actors()
        for a in actors.values():
            print(f"  [{a.threat_level}] {a.actor_id}: {a.aliases} "
                  f"({a.incident_count} incidents)")

    elif args.command == "check":
        matches = inv.resolve_identity(args.sender, discord_id=args.discord_id)
        if matches:
            for m in matches:
                print(f"  MATCH: {m.actor_id} ({m.aliases}, "
                      f"{m.incident_count} incidents)")
        else:
            print("  No matches found")

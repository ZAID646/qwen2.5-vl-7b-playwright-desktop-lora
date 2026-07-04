from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    id: str
    name: str
    start_url: str
    task: str
    expected_outcome: str
    tags: list[str] = field(default_factory=list)


SCENARIOS: list[Scenario] = [
    Scenario(id="s001", name="GitHub login", start_url="https://github.com/login",
             task="Log in with credentials", expected_outcome="User is logged in"),
    Scenario(id="s002", name="Search Wikipedia", start_url="https://en.wikipedia.org",
             task="Search for 'Artificial Intelligence'", expected_outcome="Search results page"),
    Scenario(id="s003", name="Google form fill", start_url="https://forms.google.com",
             task="Fill out a contact form", expected_outcome="Form submitted"),
    Scenario(id="s004", name="News article read",
             start_url="https://news.ycombinator.com",
             task="Open the top story and read comments",
             expected_outcome="Article + comments loaded"),
    Scenario(id="s005", name="E-commerce cart", start_url="https://www.amazon.com",
             task="Add an item to cart", expected_outcome="Item in cart"),
]

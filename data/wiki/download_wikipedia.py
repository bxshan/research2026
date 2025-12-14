# Author: Boxuan Shan + support from Claude Sonnet 4.5
#!/usr/bin/env python3
"""
BFS-based scraper for US high schools on Wikipedia.
Uses Breadth-First Search to explore categories level by level.
"""

import requests
import json
import time
from typing import Set, List, Dict
from collections import deque

# List of US states and territories
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming", "Washington, D.C.", "Puerto Rico", "Guam",
    "U.S. Virgin Islands", "American Samoa", "Northern Mariana Islands"
]

class WikipediaHighSchoolScraperBFS:
    def __init__(self):
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HighSchoolScraper/2.0 (Educational/Research Purpose; Python/requests)'
        })
        self.high_schools = {}  # Maps page_id to school info
        self.visited_categories = set()

    def determine_school_type(self, category: str) -> str:
        """Determine school type based on category name."""
        category_lower = category.lower()

        if "charter" in category_lower:
            return "charter"

        private_keywords = ["private", "independent", "parochial", "catholic", "christian",
                          "religious", "preparatory", "prep school", "academy"]
        if any(keyword in category_lower for keyword in private_keywords):
            return "private"

        public_keywords = ["public", "state", "municipal", "county", "district"]
        if any(keyword in category_lower for keyword in public_keywords):
            return "public"

        return "unknown"

    def get_page_details(self, page_ids: List[int], parent_category: str, state: str = None) -> Dict:
        """Fetch detailed information for pages."""
        if not page_ids:
            return {}

        school_type = self.determine_school_type(parent_category)
        batch_size = 50
        all_details = {}

        for i in range(0, len(page_ids), batch_size):
            batch = page_ids[i:i + batch_size]
            params = {
                "action": "query",
                "pageids": "|".join(map(str, batch)),
                "prop": "info",
                "inprop": "url",
                "format": "json"
            }

            try:
                response = self.session.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()

                if "query" in data and "pages" in data["query"]:
                    for page_id, page_data in data["query"]["pages"].items():
                        all_details[int(page_id)] = {
                            "title": page_data.get("title", ""),
                            "page_id": int(page_id),
                            "url": page_data.get("fullurl", ""),
                            "state": state,
                            "school_type": school_type,
                            "direct_parent_category": parent_category
                        }

                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching page details: {e}")

        return all_details

    def get_all_category_members(self, category: str, cmtype: str = "page|subcat") -> List[dict]:
        """Fetch all members of a category, handling pagination."""
        all_members = []
        cmcontinue = None

        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": cmtype,
                "cmlimit": "500",
                "format": "json"
            }

            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            try:
                response = self.session.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()

                if "query" in data and "categorymembers" in data["query"]:
                    all_members.extend(data["query"]["categorymembers"])

                if "continue" in data and "cmcontinue" in data["continue"]:
                    cmcontinue = data["continue"]["cmcontinue"]
                    time.sleep(0.1)
                else:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error fetching category {category}: {e}")
                break

        return all_members

    def is_high_school_article(self, title: str) -> bool:
        """Check if a title appears to be a high school article."""
        if title.startswith("Category:") or title.startswith("List of") or title.startswith("Template:"):
            return False

        high_school_keywords = [
            "High School",
            "high school",
            "Secondary School",
            "Preparatory School",
            "Prep School",
            "School"  # More lenient to catch more schools
        ]

        return any(keyword in title for keyword in high_school_keywords)

    def scrape_category_bfs(self, start_category: str, state: str = None, max_depth: int = 20):
        """
        Use BFS to scrape categories level by level.

        Args:
            start_category: Starting category name
            state: US state this belongs to
            max_depth: Maximum depth to explore
        """
        if start_category in self.visited_categories:
            return

        # Queue stores tuples: (category_name, depth)
        queue = deque([(start_category, 0)])

        print(f"  Starting BFS from: {start_category}")

        while queue:
            category, depth = queue.popleft()  # FIFO - process oldest first

            if depth > max_depth:
                print(f"  Max depth {max_depth} reached")
                continue

            if category in self.visited_categories:
                continue

            self.visited_categories.add(category)
            print(f"  {'  ' * depth}[Depth {depth}] Exploring: {category}")

            # Get all members of this category
            members = self.get_all_category_members(category)

            pages_to_fetch = []
            subcats_found = 0

            for member in members:
                title = member.get("title", "")
                page_id = member.get("pageid")
                ns = member.get("ns", 0)

                if ns == 14:  # Subcategory
                    subcats_found += 1
                    subcat_name = title.replace("Category:", "")
                    if subcat_name not in self.visited_categories:
                        queue.append((subcat_name, depth + 1))  # Add to END of queue

                elif ns == 0:  # Article
                    if self.is_high_school_article(title):
                        if page_id and page_id not in self.high_schools:
                            pages_to_fetch.append(page_id)

            # Fetch details for pages found
            if pages_to_fetch:
                details = self.get_page_details(pages_to_fetch, category, state)
                self.high_schools.update(details)
                print(f"  {'  ' * depth}→ Found {len(pages_to_fetch)} schools, {subcats_found} subcategories")
            else:
                print(f"  {'  ' * depth}→ Found 0 schools, {subcats_found} subcategories")

    def scrape_all_states(self):
        """Scrape high schools from all US states using BFS."""
        print(f"Starting BFS scrape for {len(US_STATES)} states/territories...")
        print("="*60)

        for i, state in enumerate(US_STATES, 1):
            print(f"\n[{i}/{len(US_STATES)}] Processing: {state}")
            print("-"*60)

            # Try different category name variations
            category_variations = [
                # General high school categories
                f"High schools in {state}",
                f"High schools in {state} (state)",
                # Public schools
                f"Public high schools in {state}",
                f"Public secondary schools in {state}",
                # Private schools
                f"Private high schools in {state}",
                f"Private schools in {state}",
                f"Independent schools in {state}",
                # Charter schools
                f"Charter schools in {state}",
                # Educational institutions (broader)
                f"Educational institutions in {state}",
                f"Schools in {state}",
            ]

            categories_found = 0
            for category in category_variations:
                members = self.get_all_category_members(category)
                if members:
                    categories_found += 1
                    print(f"  Found seed category: {category}")
                    self.scrape_category_bfs(category, state, max_depth=20)

            if categories_found == 0:
                print(f"  WARNING: No categories found for {state}")
            else:
                print(f"  Processed {categories_found} seed categories for {state}")

    def scrape_general_category(self):
        """Scrape the general US high schools category."""
        print("\n" + "="*60)
        print("Processing general category: High schools in the United States")
        print("="*60)

        self.scrape_category_bfs("High schools in the United States", state=None, max_depth=20)

    def save_results(self, filename: str = "us_high_schools_bfs_v2.json"):
        """Save the collected high school details to files."""
        sorted_schools = sorted(self.high_schools.values(), key=lambda x: (x["state"] or "ZZZ", x["title"]))

        # Save detailed JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sorted_schools, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(sorted_schools)} high school articles with details to {filename}")

        # Save simple text file
        txt_filename = filename.replace('.json', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            for school in sorted_schools:
                f.write(f"{school['title']}\n")

        print(f"Also saved simple list to {txt_filename}")

        # Save CSV
        csv_filename = filename.replace('.json', '.csv')
        with open(csv_filename, 'w', encoding='utf-8') as f:
            f.write("Title,Page ID,URL,State,School Type,Direct Parent Category\n")
            for school in sorted_schools:
                title = school['title'].replace('"', '""')
                url = school['url']
                state = (school['state'] or "").replace('"', '""')
                school_type = school.get('school_type', 'unknown')
                category = school['direct_parent_category'].replace('"', '""')
                f.write(f'"{title}",{school["page_id"]},"{url}","{state}","{school_type}","{category}"\n')

        print(f"Also saved as CSV to {csv_filename}")

        # Save summary
        summary_filename = filename.replace('.json', '_summary.json')
        state_counts = {}
        type_counts = {}
        for school in sorted_schools:
            state = school['state'] or "Unknown/General"
            state_counts[state] = state_counts.get(state, 0) + 1

            school_type = school.get('school_type', 'unknown')
            type_counts[school_type] = type_counts.get(school_type, 0) + 1

        summary = {
            "total_schools": len(sorted_schools),
            "total_states_processed": len(US_STATES),
            "schools_by_type": dict(sorted(type_counts.items())),
            "schools_by_state": dict(sorted(state_counts.items()))
        }

        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Also saved summary to {summary_filename}")

    def print_summary(self):
        """Print a summary of the scraping results."""
        print(f"\n{'='*60}")
        print(f"SUMMARY (BFS APPROACH)")
        print(f"{'='*60}")
        print(f"Total categories visited: {len(self.visited_categories)}")
        print(f"Total high school articles found: {len(self.high_schools)}")

        # Count by state
        state_counts = {}
        type_counts = {}
        for school in self.high_schools.values():
            state = school['state'] or "Unknown/General"
            state_counts[state] = state_counts.get(state, 0) + 1

            school_type = school.get('school_type', 'unknown')
            type_counts[school_type] = type_counts.get(school_type, 0) + 1

        print(f"Total states with schools: {len(state_counts)}")
        print(f"{'='*60}\n")

        # Show school type breakdown
        print("Schools by type:")
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        for school_type, count in sorted_types:
            percentage = (count / len(self.high_schools) * 100) if self.high_schools else 0
            print(f"  {school_type.capitalize()}: {count} schools ({percentage:.1f}%)")

        print(f"\n{'='*60}\n")

        # Show top 10 states by school count
        print("Top 10 states by number of schools:")
        sorted_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (state, count) in enumerate(sorted_states[:10], 1):
            print(f"  {i}. {state}: {count} schools")

        print(f"\n{'='*60}\n")


def main():
    """Main function to run the BFS scraper."""
    print("Wikipedia US High Schools Scraper (BFS Approach - Version 2)")
    print("="*60)

    scraper = WikipediaHighSchoolScraperBFS()

    print("Using Breadth-First Search to explore categories level-by-level")
    print("This may find more schools by exploring broader category patterns...\n")

    try:
        # First scrape all state-specific categories
        scraper.scrape_all_states()

        # Then scrape the general category
        scraper.scrape_general_category()

        scraper.print_summary()
        scraper.save_results("us_high_schools_bfs_v2.json")

    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user.")
        print(f"Partial results: {len(scraper.high_schools)} high schools found so far.")

        save = input("Save partial results? (y/n): ").strip().lower()
        if save == 'y':
            scraper.save_results("us_high_schools_bfs_v2_partial.json")

    except Exception as e:
        print(f"\nError during scraping: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

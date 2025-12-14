# Author: Boxuan Shan + support from Claude Sonnet 4.5
#!/usr/bin/env python3
"""
NCES Public School Scraper (State-by-State)

Downloads Excel files from the NCES Common Core Data school search
for each state using Selenium, then combines into a master CSV.
"""

import os
import sys
import time
import zipfile
from pathlib import Path
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# State codes (FIPS)
STATE_CODES = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05',
    'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10',
    'District of Columbia': '11', 'Florida': '12', 'Georgia': '13', 'Hawaii': '15',
    'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19',
    'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22', 'Maine': '23',
    'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27',
    'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31',
    'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35',
    'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39',
    'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44',
    'South Carolina': '45', 'South Dakota': '46', 'Tennessee': '47', 'Texas': '48',
    'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '53',
    'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56',
    'American Samoa': '60', 'Guam': '66', 'Northern Mariana Islands': '69',
    'Puerto Rico': '72', 'Virgin Islands': '78',
}

BASE_URL = 'https://nces.ed.gov/ccd/schoolsearch/school_list.asp'
DOWNLOAD_DIR = Path('public_school_downloads')
OUTPUT_DIR = Path('public_school_output')


def setup_directories():
    """Create directories."""
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Clean download directory
    for f in DOWNLOAD_DIR.glob('*'):
        if f.is_file():
            f.unlink()
    logger.info("Directories ready")


def setup_driver():
    """Setup Chrome with download preferences."""
    chrome_options = Options()
    prefs = {
        "download.default_directory": str(DOWNLOAD_DIR.absolute()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Uncomment to run headless:
    # chrome_options.add_argument("--headless")

    logger.info("Starting Chrome browser...")
    logger.info("Auto-downloading correct ChromeDriver version...")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    logger.info("Chrome browser started successfully")
    return driver


def construct_url(state_code):
    """Build search URL for state."""
    params = [
        'Search=1', 'InstName=', 'SchoolID=', 'Address=', 'City=',
        f'State={state_code}', 'Zip=', 'Miles=', 'County=',
        'PhoneAreaCode=', 'Phone=', 'DistrictName=', 'DistrictID=',
        'SchoolType=1', 'SchoolType=2', 'SchoolType=3', 'SchoolType=4',
        'SpecificSchlTypes=all', 'IncGrade=-1', 'LoGrade=-1', 'HiGrade=-1'
    ]
    return f"{BASE_URL}?{'&'.join(params)}"


def download_state_data(driver, state_name, state_code):
    """Download Excel for one state."""
    logger.info(f"Downloading {state_name} ({state_code})...")

    try:
        # Navigate to search page
        url = construct_url(state_code)
        logger.info(f"  Navigating to URL...")
        driver.get(url)
        time.sleep(0.2)

        # Check if page loaded
        logger.info(f"  Page loaded, looking for download link...")

        # Find download link - it's a JavaScript link: javascript:GetExcelFile()
        download_link = None
        try:
            # Try method 1: Find by link text
            download_link = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.LINK_TEXT, "Download Excel File"))
            )
            logger.info(f"  Found 'Download Excel File' link")
        except:
            pass

        if not download_link:
            try:
                # Try method 2: Find by XPath looking for the JavaScript function
                download_link = driver.find_element(By.XPATH, "//a[contains(@href, 'GetExcelFile')]")
                logger.info(f"  Found download link by XPath")
            except:
                pass

        if not download_link:
            try:
                # Try method 3: Find any link containing "Download" and "Excel"
                download_link = driver.find_element(By.XPATH, "//a[contains(text(), 'Download') and contains(text(), 'Excel')]")
                logger.info(f"  Found download link by text content")
            except:
                pass

        if not download_link:
            logger.warning(f"  ✗ No download link for {state_name}")
            return None

        # Track files before click
        before_files = set(DOWNLOAD_DIR.glob('*'))

        # Execute the JavaScript function directly instead of clicking
        logger.info(f"  Executing GetExcelFile() JavaScript function...")
        original_window = driver.current_window_handle

        try:
            # Try executing the GetExcelFile() function directly
            driver.execute_script("GetExcelFile();")
            logger.info(f"  Executed GetExcelFile() directly")
        except Exception as e:
            logger.warning(f"  Could not execute GetExcelFile() directly: {e}")
            # Fallback to clicking the link
            try:
                download_link.click()
            except:
                driver.execute_script("arguments[0].click();", download_link)

        # Handle the popup window (appears almost immediately)
        logger.info(f"  Waiting for popup window...")
        try:
            WebDriverWait(driver, 2).until(lambda d: len(d.window_handles) > 1)

            # Switch to the popup window
            for window_handle in driver.window_handles:
                if window_handle != original_window:
                    driver.switch_to.window(window_handle)
                    break

            logger.info(f"  Switched to popup, looking for 'Download Excel File' link...")

            # Wait for the Excel file to be generated and the download link to appear
            try:
                # Wait for "Download Excel File" link in popup
                excel_link = WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "Download Excel File"))
                )
                logger.info(f"  Found 'Download Excel File' link in popup")

                # Get the href for logging
                excel_url = excel_link.get_attribute("href")
                logger.info(f"  Excel URL: {excel_url}")

                # Click the link to download
                excel_link.click()
                logger.info(f"  Clicked 'Download Excel File' to download")
                time.sleep(0.3)  # Wait for download to start

            except Exception as e:
                logger.warning(f"  Error finding/clicking 'Download Excel File': {e}")

            # Close popup and return to main window
            driver.close()
            driver.switch_to.window(original_window)
            logger.info(f"  Closed popup, waiting for download...")

        except Exception as e:
            logger.warning(f"  No popup window: {e}")
            if driver.current_window_handle != original_window:
                driver.switch_to.window(original_window)

        # Wait for new file
        max_wait = 60
        for i in range(max_wait):
            time.sleep(0.3)
            current_files = set(DOWNLOAD_DIR.glob('*'))
            new_files = current_files - before_files

            # Filter out temp files
            complete_files = [f for f in new_files if not any(
                str(f).endswith(ext) for ext in ['.crdownload', '.tmp', '.part']
            )]

            if complete_files:
                downloaded = complete_files[0]
                # Copy and rename with state info (keep original)
                ext = downloaded.suffix
                new_name = DOWNLOAD_DIR / f"{state_name.replace(' ', '_')}_{state_code}{ext}"

                # Copy the file instead of renaming
                import shutil
                try:
                    if new_name.exists():
                        new_name.unlink()
                    logger.info(f"  Copying {downloaded} to {new_name}")
                    shutil.copy2(downloaded, new_name)
                    logger.info(f"  ✓ Downloaded: {new_name.name} (original: {downloaded.name})")
                    logger.info(f"  Verifying copy exists: {new_name.exists()}")
                except Exception as copy_error:
                    logger.error(f"  Error copying file: {copy_error}")
                return new_name

        logger.warning(f"  ✗ Timeout waiting for download for {state_name}")
        return None

    except Exception as e:
        logger.error(f"  ✗ Error for {state_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_file(filepath):
    """Extract and read Excel file."""
    try:
        # Just try to read the file - don't validate format
        # Keep HTML files for debugging

        # If zip, extract and rename
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as z:
                excel_files = [f for f in z.namelist() if f.endswith(('.xlsx', '.xls'))]
                if excel_files:
                    z.extract(excel_files[0], DOWNLOAD_DIR)
                    excel_path = DOWNLOAD_DIR / excel_files[0]

                    # Rename extracted file to match the zip name
                    state_name = filepath.stem  # Get name without .zip
                    new_excel_name = DOWNLOAD_DIR / f"{state_name}.xlsx"
                    if excel_path != new_excel_name:
                        if new_excel_name.exists():
                            new_excel_name.unlink()
                        excel_path.rename(new_excel_name)
                        excel_path = new_excel_name
                else:
                    return None
        else:
            excel_path = filepath

        # Read Excel - handle both .xlsx and .xls formats
        try:
            df = pd.read_excel(excel_path)
        except:
            # Try with xlrd engine for older .xls files
            df = pd.read_excel(excel_path, engine='xlrd')
        logger.info(f"  Read {len(df)} schools")
        return df

    except Exception as e:
        logger.error(f"  Error reading {filepath.name}: {e}")
        return None


def main():
    logger.info("="*70)
    logger.info("PUBLIC SCHOOL SCRAPER - State by State")
    logger.info("="*70)
    logger.info(f"States to download: {len(STATE_CODES)}")

    setup_directories()

    try:
        driver = setup_driver()
    except Exception as e:
        logger.error(f"Chrome setup failed: {e}")
        logger.error("Install ChromeDriver: brew install chromedriver")
        sys.exit(1)

    dataframes = []

    try:
        for idx, (state_name, state_code) in enumerate(STATE_CODES.items(), 1):
            logger.info(f"\n[{idx}/{len(STATE_CODES)}] {state_name}")

            downloaded = download_state_data(driver, state_name, state_code)

            if downloaded:
                df = process_file(downloaded)
                if df is not None and len(df) > 0:
                    dataframes.append(df)

            time.sleep(0.2)  # Be nice to server

    finally:
        driver.quit()
        logger.info("\nBrowser closed")

    # Combine all data
    logger.info(f"\n{'='*70}")
    logger.info("COMBINING DATA")
    logger.info(f"{'='*70}")
    logger.info(f"States downloaded: {len(dataframes)}/{len(STATE_CODES)}")

    if dataframes:
        master_df = pd.concat(dataframes, ignore_index=True)
        output_path = OUTPUT_DIR / 'public_schools_master.csv'
        master_df.to_csv(output_path, index=False)

        logger.info(f"\n{'='*70}")
        logger.info("SUCCESS!")
        logger.info(f"{'='*70}")
        logger.info(f"File: {output_path.absolute()}")
        logger.info(f"Total schools: {len(master_df):,}")
        logger.info(f"Columns: {len(master_df.columns)}")
    else:
        logger.error("No data downloaded")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

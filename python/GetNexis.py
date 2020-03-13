"""
Disclaimer: The author posed the following code for academic purposes and an illustration of Selenium only.
Scraping LexisNexis may be a violation of LexisNexis user policy. Use at your own legal risk.

This is a webscraper for Uni Nexis. It requires a log in and a predefined searchlink which the user wants to scrape.
Note that it is recommendable to keep the max. number of saved articles in the searchlink below 10k. This is less prone
to errors, the exact number of scraped articles is known and the program jumps in a loop which duration is estimatable.
However, the user can also scrape articles >10k which is then a while loop.
Note, adjust timers (t1, t2) which measure the time in sec. to load pages, signing in, scrolling down, retrying page
interactions etc. t1 is for short duration actions (e.g. scroll down), t2 for long (e.g. signing in).
Set display_infos to True if you want to display status infos.
"""

import time, os
from selenium import webdriver

# load credentials
from python.ConfigUser import nexis_user, nexis_pw, path_downloads, path_chromedriver, url_searchresults

# driver, url
driver = webdriver.Chrome(path_chromedriver)
driver.set_window_size(1024, 600)
driver.maximize_window()
url_signin = 'https://signin.lexisnexis.com/lnaccess/app/signin?back=https%3A%2F%2Fadvance.lexis.com%3A443%2Fnexis-uni&aci=nu'

# waiting times
t1, t2 = 1.5, 3

# display info
display_infos = True


def try_click_by_xpath(xpath, t):
    """
    enter xpath and timer, then this function tries to click the xpath, if not possible, wait time
    """
    TrytoclickFlag = False
    while TrytoclickFlag is False:
        try:
            driver.find_element_by_xpath(xpath).click()
            TrytoclickFlag = True
        except:
            time.sleep(t)


"""
Start of web scraping
"""

# open website
driver.get(url_signin)
time.sleep(t1)

# log in
if display_infos: print('logging in')
login_user_input = driver.find_element_by_xpath('/html/body/main/section/section/form/ul/div/li[1]/label/div/input')
login_user_input.send_keys(nexis_user)
login_pw_input = driver.find_element_by_xpath('/html/body/main/section/section/form/ul/div/li[2]/label/div/input')
login_pw_input.send_keys(nexis_pw)
# click button 'Sign In'
driver.find_element_by_xpath('/html/body/main/section/section/form/ul/div/li[3]/div/input[1]').submit()

# wait for page, open 'Link zu dieser Seite', click 'Weiter'
time.sleep(t2)
if display_infos: print('opening search page')
driver.get(url_searchresults)
driver.find_element_by_xpath('/html/body/main/div/div[16]/div/div/div/section/div/menu/input[1]').click()
time.sleep(t2)

number_articles_raw = driver.find_element_by_xpath('/html/body/main/div/main/div[2]/div/div[2]/header/h2/span').text
number_articles_raw = number_articles_raw.replace('News (', '').replace(')', '').replace(' ', '').replace('+','').replace('.', '')
number_articles = int(number_articles_raw)
if display_infos: print(number_articles)

# Debugging: identify how many pages and set certain help values
# number_singlepages = int(number_articles/50) + (number_articles % 50 > 0) # each 50, actual pages displayed on webpage
# number_pages = int(number_articles/100) + (number_articles % 100 > 0) # each 100, actual loops over i


"""
Regular loop over fixed number of articles if number_articles contains a number <10k. Else, do a while loop (below)
"""
if number_articles < 10000:
    k = 1
    last_download = False

    ### Loop over page results, click checkboxes, define download options and download
    for i in range(1, number_articles, 100):

        # set counters
        p1, p2 = i, i + 99
        tempfile = '{}_file_{}_{}'.format(k, p1, p2)
        if display_infos: print('downloading {}'.format(tempfile))

        # info debug
        # print('____ k={}, i={}, p1={}, p2={}'.format(k,i,p1,p2))

        # click checkbox for part 1 e.g. 1-50
        if display_infos: print('page {}-{}'.format(i, i + 49))
        try_click_by_xpath(xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[1]/input',
                           t=t2)

        # scroll down and wait for the 'Next Page' Button '>' being clickable
        time.sleep(t1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(t1)

        if number_articles - i <= 50:

            # click download
            try_click_by_xpath(
                xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[4]/ul/li[3]/button',
                t=t2)
            tempfile = '{}_file_{}_{}'.format(k, p1, p1 + number_articles - i)
            if display_infos: print('downloading final file {}'.format(tempfile))

            # set flag to indicate last downloading round
            last_download = True

        elif 50 < number_articles - i <= 100:

            # find button with triangle: > (next-page button) and click it
            driver.find_element_by_xpath('//a[@class="icon la-TriangleRight action"]').click()

            # click download
            try_click_by_xpath(
                xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[4]/ul/li[3]/button',
                t=t2)
            tempfile = '{}_file_{}_{}'.format(k, p1, p1 + number_articles - i)
            if display_infos: print('downloading final file {}'.format(tempfile))

            # set flag to indicate last downloading round
            last_download = True

        else:

            # find button with triangle: > (next-page button) and click it
            driver.find_element_by_xpath('//a[@class="icon la-TriangleRight action"]').click()

            # click checkbox for part 2 e.g. 51-100
            if display_infos: print('page {}-{}'.format(i + 49, i + 99))
            try_click_by_xpath(
                xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[1]/input', t=t1)

            # click download
            try_click_by_xpath(
                xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[4]/ul/li[3]/button',
                t=t2)

        ### Download options (need to be set up only in the first instance)
        ## 'Basis-Optionen'
        if i == 1:
            if display_infos: print('selecting options "Basis-Optionen"')
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[1]/ul/li[2]/a', t=t1)

            # click 'Basis-Optionen' > Dateityp: 'MS Word (docx)'
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[3]/div[2]/input',
                               t=t1)

            # click 'Basis-Optionen' > Beim Herunterladen mehrerer Dokumente: 'Als separater Dateien speichern'
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[4]/div[2]/input',
                               t=t1)

        # Set filename 'Basis-Optionen' > Dateiname
        try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input', t=t1)
        time.sleep(t1)
        driver.find_element_by_xpath('/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input').clear()
        time.sleep(t1)
        driver.find_element_by_xpath(
            '/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input').send_keys(tempfile)
        time.sleep(t1)

        ## 'Layout-Optionen'
        if i == 1:
            if display_infos: print('selecting options "Layout-Optionen"')
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[1]/ul/li[3]/a', t=t1)
            # click 'Eingebettete Referenzen als Links' to uncheck checkbox
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[3]/section/div[1]/fieldset/div[5]/input',
                               t=t1)

        ## Click 'Herunterladen' to download
        if display_infos: print('downloading')
        try_click_by_xpath(xpath='/html/body/aside/footer/div/button[1]', t=t1)

        # wait, check whether files were downloaded, if not, wait more
        DownloadedFlag = False
        while DownloadedFlag is False:
            time.sleep(t2)
            if os.path.isfile(path_downloads + tempfile + '.ZIP') or os.path.isfile(path_downloads + tempfile):
                if display_infos: print('file {} downloaded'.format(tempfile + '.ZIP'))
                DownloadedFlag = True
                time.sleep(t2)  # wait in case file is still downloading
            else:
                if display_infos: print('file {} not downloaded, yet. waiting more'.format(tempfile + '.ZIP'))

        # close Downloading Windows and switch back to search results
        default_handle = driver.current_window_handle
        handles = list(driver.window_handles)
        assert len(handles) > 1
        handles.remove(default_handle)
        assert len(handles) > 0
        driver.switch_to.window(handles[0])
        driver.close()
        driver.switch_to.window(default_handle)
        time.sleep(t2)

        # if very last page reached
        if last_download is True:
            if display_infos: print('last page reached, closing driver')
            driver.quit()

        # scroll down and wait for the 'Next Page' Button '>' being clickable
        time.sleep(t1)
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(t1)

        # find button with triangle: > (next-page button) and click it
        driver.find_element_by_xpath('//a[@class="icon la-TriangleRight action"]').click()

        k += 1


"""
While loop over of undetermined number of articles if number_articles contains a number >=10k.
"""

if number_articles == 10000:
    k, i = 1, 1
    next_page_clickable, last_page_50th, last_page_100th = True, False, False

    ### While Loop over page results, continue scraping if 'Next Page >' is clickable
    while next_page_clickable:

        # set counters
        p1, p2 = i, i + 99
        tempfile = '{}_file_{}_{}'.format(k, p1, p2)
        if display_infos: print('downloading {}'.format(tempfile))

        # info debug
        #print('____ k={}, i={}, p1={}, p2={}'.format(k, i, p1, p2))

        # click checkbox for part 1 e.g. 1-50
        try_click_by_xpath(xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[1]/input',
                           t=t2)

        # Get the number of selected pages
        time.sleep(t1)
        pg1 = driver.find_element_by_xpath('/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[2]/div/button/span[1]').text
        time.sleep(t1)
        pg1 = int(pg1)

        # scroll down and wait for the 'Next Page' Button '>' being clickable
        time.sleep(t1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(t1)

        ## try to click the 'Next Page' Button '>', if not available, we are on the last page
        try:
            if display_infos: print('page {}-{}'.format(i, i+pg1-1))
            # find button with triangle: > (next-page button) and click it
            driver.find_element_by_xpath('//a[@class="icon la-TriangleRight action"]').click()
            if display_infos: print('going to next page')
        except:
            next_page_clickable = False
            if display_infos: print('last page reached')
            last_page_50th = True
            # end while loop
            continue

        # click checkbox for part 2 e.g. 51-100
        try_click_by_xpath(xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[1]/input',
                           t=t1)

        # Get the number of selected pages
        time.sleep(t1)
        pg2 = driver.find_element_by_xpath(
            '/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[2]/div/button/span[1]').text
        time.sleep(t1)
        pg2 = int(pg2)

        # click download
        if display_infos: print('page {}-{}'.format(i+pg1-1, i+pg2-1))
        try_click_by_xpath(
            xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[4]/ul/li[3]/button', t=t2)

        # set up file name
        tempfile = '{}_file_{}-{}'.format(k, i, i+pg2-1)

        ### Download options (need to be set up only in the first instance)
        ## 'Basis-Optionen'
        if i == 1:
            if display_infos: print('selecting options "Basis-Optionen"')
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[1]/ul/li[2]/a', t=t1)

            # click 'Basis-Optionen' > Dateityp: 'MS Word (docx)'
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[3]/div[2]/input',
                               t=t1)

            # click 'Basis-Optionen' > Beim Herunterladen mehrerer Dokumente: 'Als separater Dateien speichern'
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[4]/div[2]/input',
                               t=t1)

        # Set filename 'Basis-Optionen' > Dateiname
        try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input', t=t1)
        time.sleep(t1)
        driver.find_element_by_xpath('/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input').clear()
        time.sleep(t1)
        driver.find_element_by_xpath(
            '/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input').send_keys(tempfile)
        time.sleep(t1)

        ## 'Layout-Optionen'
        if i == 1:
            if display_infos: print('selecting options "Layout-Optionen"')
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[1]/ul/li[3]/a', t=t1)
            # click 'Eingebettete Referenzen als Links' to uncheck checkbox
            try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[3]/section/div[1]/fieldset/div[5]/input',
                               t=t1)

        ## Click 'Herunterladen' to download
        if display_infos: print('downloading')
        try_click_by_xpath(xpath='/html/body/aside/footer/div/button[1]', t=t1)

        # wait, check whether files were downloaded, if not, wait more
        DownloadedFlag = False
        while DownloadedFlag is False:
            time.sleep(t2)
            if os.path.isfile(path_downloads + tempfile + '.ZIP') or os.path.isfile(path_downloads + tempfile):
                if display_infos: print('file {} downloaded'.format(tempfile + '.ZIP'))
                DownloadedFlag = True
                time.sleep(t2)  # wait in case file is still downloading
            else:
                if display_infos: print('file {} not downloaded, yet. waiting more'.format(tempfile + '.ZIP'))

        # close Downloading Windows and switch back to search results
        default_handle = driver.current_window_handle
        handles = list(driver.window_handles)
        assert len(handles) > 1
        handles.remove(default_handle)
        assert len(handles) > 0
        driver.switch_to.window(handles[0])
        driver.close()
        driver.switch_to.window(default_handle)
        time.sleep(t2)

        # scroll down and wait for the 'Next Page' Button '>' being clickable
        time.sleep(t1)
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(t1)

        ## try to click the 'Next Page' Button '>', if not available, this was the last page
        try:
            driver.find_element_by_xpath('//a[@class="icon la-TriangleRight action"]').click()
            if display_infos: print('going to next page')
        except:
            next_page_clickable = False
            if display_infos: print('last page reached')
            # end while loop
            continue

        k += 1
        i += 100

    ## if very last page reached, download rest and close driver
    if display_infos: print('last page reached, closing driver')
    k += 1
    i += 100

    # if we are on the 50th page type
    if last_page_50th:
        # click download
        try_click_by_xpath(
            xpath='/html/body/main/div/main/div[2]/div/div[2]/div[2]/form/div[1]/div/ul[1]/li[4]/ul/li[3]/button', t=t2)
        # Set filename 'Basis-Optionen' > Dateiname
        try_click_by_xpath(xpath='/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input', t=t1)
        # set filename for last download
        tempfile = '{}_file_{}_{}'.format(k, i, i+pg1-1)
        if display_infos: print('downloading last file {}'.format(tempfile))

        time.sleep(t1)
        driver.find_element_by_xpath('/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input').clear()
        time.sleep(t1)
        driver.find_element_by_xpath(
            '/html/body/aside/form/div[4]/div[2]/div[2]/section/fieldset[6]/div/input').send_keys(tempfile)
        time.sleep(t1)

    if display_infos: print('closing driver')
    driver.quit()

    print('DONE!')

import numbers
from datetime import datetime
import re

import numpy as np
import pandas as pd

country_codes = {
    'AD': 'Andorra',
    'AE': 'United Arab Emirates',
    'AF': 'Afghanistan',
    'AG': 'Antigua & Barbuda',
    'AI': 'Anguilla',
    'AL': 'Albania',
    'AM': 'Armenia',
    'AN': 'Netherlands Antilles',
    'AO': 'Angola',
    'AQ': 'Antarctica',
    'AR': 'Argentina',
    'AS': 'American Samoa',
    'AT': 'Austria',
    'AU': 'Australia',
    'AW': 'Aruba',
    'AZ': 'Azerbaijan',
    'BA': 'Bosnia and Herzegovina',
    'BB': 'Barbados',
    'BD': 'Bangladesh',
    'BE': 'Belgium',
    'BF': 'Burkina Faso',
    'BG': 'Bulgaria',
    'BH': 'Bahrain',
    'BI': 'Burundi',
    'BJ': 'Benin',
    'BM': 'Bermuda',
    'BN': 'Brunei Darussalam',
    'BO': 'Bolivia',
    'BR': 'Brazil',
    'BS': 'Bahama',
    'BT': 'Bhutan',
    'BU': 'Burma (no longer exists)',
    'BV': 'Bouvet Island',
    'BW': 'Botswana',
    'BY': 'Belarus',
    'BZ': 'Belize',
    'CA': 'Canada',
    'CC': 'Cocos (Keeling) Islands',
    'CF': 'Central African Republic',
    'CG': 'Congo',
    'CH': 'Switzerland',
    'CI': 'Côte D\'ivoire (Ivory Coast)',
    'CK': 'Cook Iislands',
    'CL': 'Chile',
    'CM': 'Cameroon',
    'CN': 'China',
    'CO': 'Colombia',
    'CR': 'Costa Rica',
    'CS': 'Czechoslovakia (no longer exists)',
    'CU': 'Cuba',
    'CV': 'Cape Verde',
    'CX': 'Christmas Island',
    'CY': 'Cyprus',
    'CZ': 'Czech Republic',
    'DD': 'German Democratic Republic (no longer exists)',
    'DE': 'Germany',
    'DJ': 'Djibouti',
    'DK': 'Denmark',
    'DM': 'Dominica',
    'DO': 'Dominican Republic',
    'DZ': 'Algeria',
    'EC': 'Ecuador',
    'EE': 'Estonia',
    'EG': 'Egypt',
    'EH': 'Western Sahara',
    'ER': 'Eritrea',
    'ES': 'Spain',
    'ET': 'Ethiopia',
    'FI': 'Finland',
    'FJ': 'Fiji',
    'FK': 'Falkland Islands (Malvinas)',
    'FM': 'Micronesia',
    'FO': 'Faroe Islands',
    'FR': 'France',
    'FX': 'France, Metropolitan',
    'GA': 'Gabon',
    'GB': 'United Kingdom (Great Britain)',
    'GD': 'Grenada',
    'GE': 'Georgia',
    'GF': 'French Guiana',
    'GH': 'Ghana',
    'GI': 'Gibraltar',
    'GL': 'Greenland',
    'GM': 'Gambia',
    'GN': 'Guinea',
    'GP': 'Guadeloupe',
    'GQ': 'Equatorial Guinea',
    'GR': 'Greece',
    'GS': 'South Georgia and the South Sandwich Islands',
    'GT': 'Guatemala',
    'GU': 'Guam',
    'GW': 'Guinea-Bissau',
    'GY': 'Guyana',
    'HK': 'Hong Kong',
    'HM': 'Heard & McDonald Islands',
    'HN': 'Honduras',
    'HR': 'Croatia',
    'HT': 'Haiti',
    'HU': 'Hungary',
    'ID': 'Indonesia',
    'IE': 'Ireland',
    'IL': 'Israel',
    'IN': 'India',
    'IO': 'British Indian Ocean Territory',
    'IQ': 'Iraq',
    'IR': 'Islamic Republic of Iran',
    'IS': 'Iceland',
    'IT': 'Italy',
    'JM': 'Jamaica',
    'JO': 'Jordan',
    'JP': 'Japan',
    'KE': 'Kenya',
    'KG': 'Kyrgyzstan',
    'KH': 'Cambodia',
    'KI': 'Kiribati',
    'KM': 'Comoros',
    'KN': 'St. Kitts and Nevis',
    'KP': 'Korea, Democratic People\'s Republic of',
    'KR': 'Korea, Republic of',
    'KW': 'Kuwait',
    'KY': 'Cayman Islands',
    'KZ': 'Kazakhstan',
    'LA': 'Lao People\'s Democratic Republic',
    'LB': 'Lebanon',
    'LC': 'Saint Lucia',
    'LI': 'Liechtenstein',
    'LK': 'Sri Lanka',
    'LR': 'Liberia',
    'LS': 'Lesotho',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'LV': 'Latvia',
    'LY': 'Libyan Arab Jamahiriya',
    'MA': 'Morocco',
    'MC': 'Monaco',
    'MD': 'Moldova, Republic of',
    'MG': 'Madagascar',
    'MH': 'Marshall Islands',
    'ML': 'Mali',
    'MN': 'Mongolia',
    'MM': 'Myanmar',
    'MO': 'Macau',
    'MP': 'Northern Mariana Islands',
    'MQ': 'Martinique',
    'MR': 'Mauritania',
    'MS': 'Monserrat',
    'MT': 'Malta',
    'MU': 'Mauritius',
    'MV': 'Maldives',
    'MW': 'Malawi',
    'MX': 'Mexico',
    'MY': 'Malaysia',
    'MZ': 'Mozambique',
    'NA': 'Namibia',
    'NC': 'New Caledonia',
    'NE': 'Niger',
    'NF': 'Norfolk Island',
    'NG': 'Nigeria',
    'NI': 'Nicaragua',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'NP': 'Nepal',
    'NR': 'Nauru',
    'NT': 'Neutral Zone (no longer exists)',
    'NU': 'Niue',
    'NZ': 'New Zealand',
    'OM': 'Oman',
    'PA': 'Panama',
    'PE': 'Peru',
    'PF': 'French Polynesia',
    'PG': 'Papua New Guinea',
    'PH': 'Philippines',
    'PK': 'Pakistan',
    'PL': 'Poland',
    'PM': 'St. Pierre & Miquelon',
    'PN': 'Pitcairn',
    'PR': 'Puerto Rico',
    'PT': 'Portugal',
    'PW': 'Palau',
    'PY': 'Paraguay',
    'QA': 'Qatar',
    'RE': 'Réunion',
    'RO': 'Romania',
    'RU': 'Russian Federation',
    'RW': 'Rwanda',
    'SA': 'Saudi Arabia',
    'SB': 'Solomon Islands',
    'SC': 'Seychelles',
    'SD': 'Sudan',
    'SE': 'Sweden',
    'SG': 'Singapore',
    'SH': 'St. Helena',
    'SI': 'Slovenia',
    'SJ': 'Svalbard & Jan Mayen Islands',
    'SK': 'Slovakia',
    'SL': 'Sierra Leone',
    'SM': 'San Marino',
    'SN': 'Senegal',
    'SO': 'Somalia',
    'SR': 'Suriname',
    'ST': 'Sao Tome & Principe',
    'SU': 'Union of Soviet Socialist Republics (no longer exists)',
    'SV': 'El Salvador',
    'SY': 'Syrian Arab Republic',
    'SZ': 'Swaziland',
    'TC': 'Turks & Caicos Islands',
    'TD': 'Chad',
    'TF': 'French Southern Territories',
    'TG': 'Togo',
    'TH': 'Thailand',
    'TJ': 'Tajikistan',
    'TK': 'Tokelau',
    'TM': 'Turkmenistan',
    'TN': 'Tunisia',
    'TO': 'Tonga',
    'TP': 'East Timor',
    'TR': 'Turkey',
    'TT': 'Trinidad & Tobago',
    'TV': 'Tuvalu',
    'TW': 'Taiwan, Province of China',
    'TZ': 'Tanzania, United Republic of',
    'UA': 'Ukraine',
    'UG': 'Uganda',
    'UM': 'United States Minor Outlying Islands',
    'US': 'United States of America',
    'UY': 'Uruguay',
    'UZ': 'Uzbekistan',
    'VA': 'Vatican City State (Holy See)',
    'VC': 'St. Vincent & the Grenadines',
    'VE': 'Venezuela',
    'VG': 'British Virgin Islands',
    'VI': 'United States Virgin Islands',
    'VN': 'Viet Nam',
    'VU': 'Vanuatu',
    'WF': 'Wallis & Futuna Islands',
    'WS': 'Samoa',
    'YD': 'Democratic Yemen (no longer exists)',
    'YE': 'Yemen',
    'YT': 'Mayotte',
    'YU': 'Yugoslavia',
    'ZA': 'South Africa',
    'ZM': 'Zambia',
    'ZR': 'Zaire',
    'ZW': 'Zimbabwe',
    'ZZ': 'Unknown or unspecified country',
}
set_of_currencies = {'OMR', 'LKR', 'HUF', 'CNY', 'QAR', 'IDR', 'JPY', 'TRY', 'AED', 'NZD', 'RON', 'NGN', 'MXN', 'CZK',
                     'UAH', 'BDT', 'NOK', 'DKK', 'USD', 'FJD', 'EGP', 'ARS', 'SEK', 'CAD', 'KWD', 'LAK', 'CHF', 'PKR',
                     'MYR', 'GBP', 'HKD', 'PLN', 'PHP', 'JOD', 'KRW', 'SAR', 'VND', 'RUB', 'SGD', 'ILS', 'BHD', 'BRL',
                     'AUD', 'KHR', 'THB', 'XPF', 'INR', 'ZAR', 'EUR', 'TWD', 'KZT'}
currency_to_dollar = {
    "AED": 3.673181,
    "AFN": 72.823258,
    "ALL": 110.58,
    "AMD": 482.840272,
    "ANG": 1.843953,
    "AOA": 270.3925,
    "ARS": 29.724,
    "AUD": 1.3776,
    "AWG": 1.7925,
    "AZN": 1.7025,
    "BAM": 1.72015,
    "BBD": 2,
    "BDT": 84.453473,
    "BGN": 1.71989,
    "BHD": 0.377088,
    "BIF": 1772.564836,
    "BMD": 1,
    "BND": 1.51076,
    "BOB": 6.90853,
    "BRL": 3.905406,
    "BSD": 1,
    "BTC": 0.000158408561,
    "BTN": 70.250498,
    "BWP": 10.8055,
    "BYN": 2.0534,
    "BZD": 2.008866,
    "CAD": 1.31605,
    "CDF": 1626.551914,
    "CHF": 0.997384,
    "CLF": 0.02338,
    "CLP": 669.261521,
    "CNH": 6.868637,
    "CNY": 6.88415,
    "COP": 3047.975024,
    "CRC": 567.12861,
    "CUC": 1,
    "CUP": 25.5,
    "CVE": 97.3,
    "CZK": 22.6324,
    "DJF": 178,
    "DKK": 6.557082,
    "DOP": 49.876505,
    "DZD": 119.029816,
    "EGP": 17.902,
    "ERN": 14.9965,
    "ETB": 27.568,
    "EUR": 0.879356,
    "FJD": 2.116905,
    "FKP": 0.786624,
    "GBP": 0.786624,
    "GEL": 2.482717,
    "GGP": 0.786624,
    "GHS": 4.874117,
    "GIP": 0.786624,
    "GMD": 48.16,
    "GNF": 9036.151169,
    "GTQ": 7.4894,
    "GYD": 209.103315,
    "HKD": 7.84972,
    "HNL": 24.03,
    "HRK": 6.5276,
    "HTG": 67.3475,
    "HUF": 284.843812,
    "IDR": 14344.516583,
    "ILS": 3.67127,
    "IMP": 0.786624,
    "INR": 70.015,
    "IQD": 1191.56269,
    "IRR": 43163.26868,
    "ISK": 108.329847,
    "JEP": 0.786624,
    "JMD": 135.5825,
    "JOD": 0.709503,
    "JPY": 111.01484,
    "KES": 100.790129,
    "KGS": 68.137481,
    "KHR": 4070.169979,
    "KMF": 432.952376,
    "KPW": 900,
    "KRW": 1127.32,
    "KWD": 0.303322,
    "KYD": 0.832898,
    "KZT": 360.032943,
    "LAK": 8518.942724,
    "LBP": 1511,
    "LKR": 160.413162,
    "LRD": 154.549609,
    "LSL": 14.255,
    "LYD": 1.392443,
    "MAD": 9.5652,
    "MDL": 16.632113,
    "MGA": 3324.709248,
    "MKD": 54.135,
    "MMK": 1527.15,
    "MNT": 2442.166667,
    "MOP": 8.0806,
    "MRO": 357.5,
    "MRU": 35.95,
    "MUR": 34.850029,
    "MVR": 15.459996,
    "MWK": 727.203141,
    "MXN": 18.998056,
    "MYR": 4.1055,
    "MZN": 58.989229,
    "NAD": 14.537382,
    "NGN": 361.020294,
    "NIO": 31.871276,
    "NOK": 8.481203,
    "NPR": 112.403423,
    "NZD": 1.518912,
    "OMR": 0.38496,
    "PAB": 1,
    "PEN": 3.312,
    "PGK": 3.311548,
    "PHP": 53.425938,
    "PKR": 122.755692,
    "PLN": 3.78625,
    "PYG": 5750.35,
    "QAR": 3.641064,
    "RON": 4.0961,
    "RSD": 103.760733,
    "RUB": 66.8698,
    "RWF": 877.665,
    "SAR": 3.7507,
    "SBD": 7.88911,
    "SCR": 13.588838,
    "SDG": 17.990205,
    "SEK": 9.194411,
    "SGD": 1.37585,
    "SHP": 0.786624,
    "SLL": 6542.71,
    "SOS": 578.345,
    "SRD": 7.458,
    "SSP": 130.2634,
    "STD": 21050.59961,
    "STN": 21.575,
    "SVC": 8.745692,
    "SYP": 514.97999,
    "SZL": 14.537219,
    "THB": 33.195021,
    "TJS": 9.419687,
    "TMT": 3.509961,
    "TND": 2.769793,
    "TOP": 2.310538,
    "TRY": 5.852199,
    "TTD": 6.736358,
    "TWD": 30.76086,
    "TZS": 2286.489273,
    "UAH": 27.619953,
    "UGX": 3751.002075,
    "USD": 1,
    "UYU": 31.573521,
    "UZS": 7791.674003,
    "VEF": 141572.666667,
    "VND": 23114.085172,
    "VUV": 108.499605,
    "WST": 2.588533,
    "XAF": 576.819847,
    "XAG": 0.06772851,
    "XAU": 0.00084235,
    "XCD": 2.70255,
    "XDR": 0.717117,
    "XOF": 576.819847,
    "XPD": 0.00101,
    "XPF": 104.935106,
    "XPT": 0.00127078,
    "YER": 250.3,
    "ZAR": 14.56358,
    "ZMW": 10.218987,
    "ZWL":322.355011
}
def preprocess(filename, mode=0):

    df = pd.read_csv(filename)

    df['hotel_star_rating'] = df['hotel_star_rating'].apply(lambda x: 3.5 if (not isinstance(x, numbers.Number) or x < 1 or x > 5) else int(x))
    df['guest_is_not_the_customer'] = df['guest_is_not_the_customer'].apply(lambda x: 0 if (not isinstance(x, numbers.Number) or not x == 1) else x)
    df['no_of_adults'] = df['no_of_adults'].apply(lambda x: 2 if(not isinstance(x, numbers.Number) or x < 1) else int(x))
    df['no_of_children'] = df['no_of_children'].apply(lambda x: 0 if (not isinstance(x, numbers.Number) or x < 1) else int(x))
    df['no_of_room'] = df['no_of_room'].apply(lambda x: 1 if (not isinstance(x, numbers.Number) or x < 1) else int(x))
    df['no_of_extra_bed'] = df['no_of_extra_bed'].apply(lambda x: 1 if (not isinstance(x, numbers.Number) or x < 1) else int(x))
    df['original_selling_amount'] = df['original_selling_amount'].apply(lambda x: 200 if (not isinstance(x, numbers.Number) or x < 0) else int(x))

    df['request_airport'] = df['request_airport'].apply(lambda x: 0 if (not isinstance(x, numbers.Number) or not x == 1) else x)
    df['request_nonesmoke'] = df['request_nonesmoke'].apply(
        lambda x: 0 if (not isinstance(x, numbers.Number) or not x == 1) else x)
    df['request_latecheckin'] = df['request_latecheckin'].apply(
        lambda x: 0 if (not isinstance(x, numbers.Number) or not x == 1) else x)
    df['request_highfloor'] = df['request_highfloor'].apply(
        lambda x: 0 if (not isinstance(x, numbers.Number) or not x == 1) else x)
    df['request_earlycheckin'] = df['request_earlycheckin'].apply(
        lambda x: 0 if (not isinstance(x, numbers.Number) or not x == 1) else x)

    df['booking_hour'] = df['booking_datetime'].apply(lambda x:1 if (extract_hour(x) > 23 or extract_hour(x) < 7) else 0)
    df['days_ahead'] = df.apply(lambda row: calculate_date_difference(row['booking_datetime'], row['checkin_date']), axis=1)
    df.loc[(df['days_ahead'] == -1), 'days_ahead'] = 30
    df['same_day_booking_checkin'] = df['days_ahead'].apply(lambda x: 1 if x==0 else 0)

    df['vacation_days_duration'] = df.apply(lambda row: calculate_date_difference(row['checkin_date'], row['checkout_date']), axis=1)
    df.loc[(df['vacation_days_duration'] == -1), 'vacation_days_duration'] = 2

    df['pay_later'] = df['charge_option'].apply(lambda x: 0 if (not x in ['Pay Later', 'Pay at Check-in']) else 1)
    df['credit'] = df['original_payment_type'].apply(lambda x: 0 if x != 'Credit Card' else 1)

    df['in_country'] = df.apply( lambda row: check_in_country_vacation(row['hotel_country_code'], row['customer_nationality']) , axis=1)
    df['is_user_logged_in'] = df['is_user_logged_in'].apply(lambda x:1 if x is True else 0)

    for col in ['>30', '8-30', '4-7', '2-3', '<1','>30_', '8-30_', '4-7_', '2-3_', '<1_']:
        df[col] = np.zeros(len(df))
    df['other_currency'] = [0] * len(df)
    for cur in set_of_currencies:
        df[cur] = np.zeros(len(df))


    for index, row in df.iterrows():
        df.at[index, 'original_selling_amount'] = row['original_selling_amount'] * currency_to_dollar[
            row['original_payment_currency']]
        policy = extract_cancellation_policy(df.at[index, 'cancellation_policy_code'], df.at[index, 'vacation_days_duration'], df.at[index, 'original_selling_amount'])

        for key in policy.keys():
            value = policy[key]
            if key <= 1:
                df.at[index, '<1'] = max(df.at[index, '<1'],value)
                df.at[index, '<1_'] = max(df.at[index, '<1_'], value/(max(row['original_selling_amount'],value)))
            elif key <=3:
                df.at[index, '2-3'] = max(df.at[index, '2-3'],value)
                df.at[index, '2-3_'] = max(df.at[index, '2-3_'], value / (max(row['original_selling_amount'], value)))
            elif key <= 7:
                df.at[index, '4-7'] = max(df.at[index, '4-7'],value)
                df.at[index, '4-7_'] = max(df.at[index, '4-7_'], value / (max(row['original_selling_amount'], value)))
            elif key <= 30:
                df.at[index, '8-30'] = max(df.at[index, '8-30'],value)
                df.at[index, '8-30_'] = max(df.at[index, '8-30_'], value / (max(row['original_selling_amount'], value)))
            else:
                df.at[index, '>30'] = max(df.at[index, '>30'],value)
                df.at[index, '>30_'] = max(df.at[index, '>30_'], value / (max(row['original_selling_amount'], value)))

        vals = ['>30', '8-30', '4-7', '2-3', '<1']
        for i,r in enumerate(vals):
            if df.at[index, r] == 0 and i >0:
                df.at[index, r] = df.at[index, vals[i-1]]

        vals = ['>30_', '8-30_', '4-7_', '2-3_', '<1_']
        for i, r in enumerate(vals):
            if df.at[index, r] == 0 and i > 0:
                df.at[index, r] = df.at[index, vals[i - 1]]

        if row['original_payment_currency'] in set_of_currencies:
            df.at[index, row['original_payment_currency']] = 1
        else:
            df.at[index, 'other_currency'] = 1


    bins = [0, 50,100, 500, 1000, 5000, 10000, float('inf')]
    labels = ['<50', '50-100', '100-500', '500-1000', '1000-5000', '5000-10000', '>10000']
    df['selling_amount_range'] = pd.cut(df['original_selling_amount'], bins=bins, labels=labels)
    dummy = pd.get_dummies(df['selling_amount_range'], prefix='selling_amount_range ')
    df = pd.concat([df, dummy], axis=1)


    if mode != 1:
        df = df.drop('h_booking_id', axis=1)

    columns_to_remove = [ 'hotel_id', 'h_customer_id', 'hotel_live_date',
                          'request_largebed', 'request_twinbeds', 'hotel_area_code',
                         'hotel_brand_code', 'hotel_chain_code', 'hotel_city_code', 'booking_datetime', 'checkin_date', 'checkout_date',
                         'hotel_country_code', 'hotel_live_date', 'accommadation_type_name', 'guest_nationality_country_name',
                         'origin_country_code', 'language', 'original_payment_method', 'original_payment_type', 'charge_option',
                         'is_first_booking', 'original_selling_amount', 'cancellation_policy_code', 'customer_nationality',
                         'booking_hour', 'selling_amount_range','original_payment_currency']
    df = df.drop(columns_to_remove, axis=1)

    ##cancelation field
    if 'cancellation_datetime' in df.columns:
        df['cancellation_datetime'] = df['cancellation_datetime'].isna()
        df['cancellation_datetime'] = df['cancellation_datetime'].apply(lambda x: 0 if x is True else 1)
        pass

    #df.to_csv("clean_train.csv", index=False)
    return df




def extract_hour(date_string):
    try:
        # Convert the date string to a datetime object
        date_obj = datetime.strptime(date_string, '%d/%m/%Y %H:%M:%S')
        # Extract the hour from the datetime object
        hour = date_obj.hour
    except (ValueError, TypeError):
        # Handle cases of invalid or non-string inputs by assigning a default hour
        hour = 14
    return hour

def calculate_date_difference(date_string1, date_string2):
    format_string = '%Y-%m-%d %H:%M:%S'
    try:
        date1 = datetime.strptime(date_string1, format_string).replace(hour=0, minute=0, second=0)
        date2 = datetime.strptime(date_string2, format_string).replace(hour=0, minute=0, second=0)
        difference = date2 - date1

    except (ValueError, TypeError):
        difference = -1

    return difference.days if difference.days >= 0 else -1

def extract_cancellation_policy(policy, nights, total_payment):
    cancellation_conditions = {}

    # Extract days before check-in and charge pattern
    pattern = r'(\d+)D((\d+)[PN])'
    matches = re.findall(pattern, policy)
    if matches:
        for match in matches:
            days_before_checkin = int(match[0])
            charge = match[2]
            if 'N' in match[1]:
                charge = float(charge)* (total_payment/nights)
            else:
                charge = (float(charge)/100)*total_payment
            cancellation_conditions[int(days_before_checkin)] = charge

    # Extract no-show charge pattern
    pattern = r'(\d+[PN])$'
    match = re.search(pattern, policy)
    if match:
        no_show_charge = match.group(1)
        if 'N' in match[1]:
            cancellation_conditions[0] = float(int(no_show_charge[:-1])) * (total_payment / nights)
        else:
            cancellation_conditions[0] = (float(int(no_show_charge[:-1])) / 100) * total_payment

    if len(cancellation_conditions) == 0:
        cancellation_conditions[31] = total_payment

    return cancellation_conditions

def check_in_country_vacation(dest, src):
    if country_codes.keys().__contains__(dest) and country_codes[dest].lower() == src.lower():
        return 1
    return 0



if __name__ == "__main__":
    preprocess('agoda_cancellation_train.csv')
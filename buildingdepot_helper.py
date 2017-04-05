import requests
import time
import json
import time
import calendar

class BuildingDepotHelper:
    def __init__(self):
        self.bd_rest_api = 'https://bd-exp.andrew.cmu.edu'
        self.client_id = 'EzjOE9jOdsKkahajC18omoFLYsxXKhRuul2SS9se'
        self.secret_key = 'DYT6nG7J01nCamDjUxUMw7TV5FY1Rb406C4QUBTWOZoGsDLxtG'
        self.access_token = self.get_oauth_token()
	

    def get_oauth_token(self):
        headers = {'content-type': 'application/json'}
        url = self.bd_rest_api
        url += ':81' 
        url += '/oauth/access_token/client_id='
        url += self.client_id
        url += '/client_secret='
        url += self.secret_key
        result = requests.get(url, headers=headers)
	
        if result.status_code == 200:
            dic = result.json()
            return dic['access_token']
        else:
            return ''

    def get_timeseries_data(self, uuid, start_time, end_time):
        headers = {
            'content-type': 'application/json',
            'Authorization': 'Bearer ' + self.access_token
            }
        url = self.bd_rest_api
        url += ':82' 
        url += '/api/sensor/'
        url += uuid + '/timeseries?'
        url += 'start_time=' + str(start_time)
        url += '&end_time=' + str(end_time)
	

        result = requests.get(url, headers=headers)
        json = result.json()

	if not 'series' in json['data']:
		return []
	
        readings = json['data']['series'][0]

        columns = readings['columns']
        values = readings['values']
        index = columns.index('value')

        data = []
        for value in values:
            data.append(value[index])

        return data


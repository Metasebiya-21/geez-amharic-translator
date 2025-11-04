import json
import requests



class NotifyCustomer:
    def __init__(self) -> None:
        self.message = "ሰላም {full_name}፣ እንኳን ደስ ያለዎት! አሁን ለማለዳ ብድር ብቁ ነዎት። የአበዳሪ መተግበሪያውን እዚህ ያውርዱ፡ {link}"

    def _load_json(self, fpath):
        with open(fpath, 'r') as f:
            return json.load(f)
    
    def send_notification(self, customer_list, link):
        customers = self._load_json(customer_list)
        for customer  in customers:
            message = self.message.format(
                full_name=customer['full_name'],
                link=link
            )
    
    def send_sms(self, phone, message):
        url = "http://196.189.181.102:48082/SmsNotification/api/message"

        payload = json.dumps({
        "message": message,
        "phoneNo": phone,
        "tokenId": "1402513020912170310287"
        })
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

if __name__ == '__main__':
    notifier = NotifyCustomer()
    notifier.send_notification(
        customer_list="",
        link = "https://play.google.com/store/apps/details?id=com.qena.abol&hl=en"
    )
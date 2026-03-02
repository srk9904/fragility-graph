import boto3
import time
import logging
from datetime import datetime, timedelta
from .config import settings

logger = logging.getLogger(__name__)

class CostMonitor:
    def __init__(self):
        try:
            self.client = boto3.client(
                'ce', # Cost Explorer
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID if settings.AWS_ACCESS_KEY_ID != "dummy" else None,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if settings.AWS_SECRET_ACCESS_KEY != "dummy" else None,
                region_name='us-east-1' # Cost Explorer is global but usually accessed via us-east-1
            )
            self.last_alert_threshold = 0
            self.max_cap = 275.0
        except Exception as e:
            logger.error(f"Failed to initialize Cost Explorer client: {e}")
            self.client = None

    def check_costs(self):
        if not self.client:
            return

        try:
            # Query for costs from the first of the month until today
            start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            response = self.client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='MONTHLY',
                Metrics=['UnblendedCost']
            )

            current_cost = float(response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])
            
            logger.info(f"Current AWS Spend: ${current_cost:.2f}")

            # Alert logic
            if current_cost >= self.max_cap:
                self._trigger_big_alert("CRITICAL: AWS CAP REACHED", f"Current cost ${current_cost:.2f} exceeds cap ${self.max_cap:.2f}")
            
            # Alert every $50
            threshold = int(current_cost // 50) * 50
            if threshold > self.last_alert_threshold:
                self._trigger_big_alert("BUDGET ALERT", f"You have reached a new $50 milestone! Total spend: ${current_cost:.2f}")
                self.last_alert_threshold = threshold

        except Exception as e:
            logger.error(f"Error checking AWS costs: {e}")

    def _trigger_big_alert(self, title, message):
        """
        Creates a high-visibility terminal alert.
        """
        border = "*" * 60
        alert = f"""
{border}
{title.center(60)}
{border}
{message.center(60)}
{border}
        """
        print(alert)
        logger.warning(alert)

cost_monitor = CostMonitor()

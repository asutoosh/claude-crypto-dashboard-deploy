import pandas as pd
import requests
import time
import json
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import schedule
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration for the alert system"""
    sheet_url: str
    whatsapp_api_token: str
    whatsapp_phone_number: str
    target_phone_number: str
    alert_threshold_percent: float = 1.0  # 1% threshold
    check_interval_minutes: int = 1
    cache_duration_minutes: int = 2

class WhatsAppNotifier:
    """Handle WhatsApp notifications using WhatsApp Business API"""
    
    def __init__(self, api_token: str, phone_number: str):
        self.api_token = api_token
        self.phone_number = phone_number
        self.base_url = f"https://graph.facebook.com/v18.0/{phone_number}/messages"
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def send_message(self, to_number: str, message: str) -> bool:
        """Send WhatsApp message"""
        try:
            data = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "text",
                "text": {"body": message}
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"WhatsApp message sent successfully to {to_number}")
                return True
            else:
                logger.error(f"Failed to send WhatsApp message: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {str(e)}")
            return False

class PriceCache:
    """Simple in-memory price cache with TTL"""
    
    def __init__(self, ttl_minutes: int = 2):
        self.cache = {}
        self.ttl_seconds = ttl_minutes * 60
    
    def get(self, symbol: str) -> Optional[float]:
        """Get cached price if still valid"""
        if symbol in self.cache:
            price, timestamp = self.cache[symbol]
            if time.time() - timestamp < self.ttl_seconds:
                return price
        return None
    
    def set(self, symbol: str, price: float):
        """Cache price with current timestamp"""
        self.cache[symbol] = (price, time.time())
    
    def clear_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]

class CryptoAlertSystem:
    """Main alert system class"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.whatsapp = WhatsAppNotifier(
            config.whatsapp_api_token,
            config.whatsapp_phone_number
        )
        self.price_cache = PriceCache(config.cache_duration_minutes)
        self.last_alerts = {}  # Track last alert time to avoid spam
        self.alert_cooldown = 300  # 5 minutes cooldown per symbol
    
    def load_sheet_data(self) -> Optional[pd.DataFrame]:
        """Load data from Google Sheet CSV URL"""
        try:
            df = pd.read_csv(self.config.sheet_url)
            logger.info(f"Loaded {len(df)} rows from Google Sheet")
            return df
        except Exception as e:
            logger.error(f"Error loading sheet data: {str(e)}")
            return None
    
    def get_batch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch multiple prices in batch from Binance"""
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            all_prices = response.json()
            
            price_dict = {}
            for symbol in symbols:
                # Check cache first
                cached_price = self.price_cache.get(symbol)
                if cached_price is not None:
                    price_dict[symbol] = cached_price
                    continue
                
                # Get from API response
                clean_symbol = symbol.replace('/', '').replace('-', '').upper() + "USDT"
                for ticker in all_prices:
                    if ticker['symbol'] == clean_symbol:
                        price = float(ticker['lastPrice'])
                        price_dict[symbol] = price
                        self.price_cache.set(symbol, price)
                        break
            
            logger.info(f"Fetched prices for {len(price_dict)} symbols")
            return price_dict
            
        except Exception as e:
            logger.error(f"Error fetching batch prices: {str(e)}")
            return {}
    
    def parse_entries(self, row: pd.Series) -> List[float]:
        """Parse entry prices from row"""
        entries = []
        entry_columns = ['Entry 1', 'Entry 2', 'Entry 3', '1st entry', '2nd entry', '3rd entry']
        
        for col in entry_columns:
            if col in row.index:
                try:
                    entry = float(row[col]) if pd.notna(row[col]) and row[col] != '' else None
                    if entry and entry > 0:
                        entries.append(entry)
                except (ValueError, TypeError):
                    continue
        
        return entries
    
    def check_entry_proximity(self, current_price: float, entries: List[float], 
                            threshold_percent: float) -> List[Tuple[float, float]]:
        """Check if current price is within threshold of any entry"""
        alerts = []
        
        for entry in entries:
            # Calculate percentage difference
            diff_percent = abs((current_price - entry) / entry) * 100
            
            if diff_percent <= threshold_percent:
                alerts.append((entry, diff_percent))
        
        return alerts
    
    def should_send_alert(self, symbol: str) -> bool:
        """Check if we should send alert (avoid spam)"""
        current_time = time.time()
        
        if symbol in self.last_alerts:
            time_since_last = current_time - self.last_alerts[symbol]
            if time_since_last < self.alert_cooldown:
                return False
        
        self.last_alerts[symbol] = current_time
        return True
    
    def format_alert_message(self, symbol: str, current_price: float, 
                           alerts: List[Tuple[float, float]]) -> str:
        """Format WhatsApp alert message"""
        message = f"🚨 CRYPTO ENTRY ALERT 🚨\n\n"
        message += f"Symbol: {symbol}\n"
        message += f"Current Price: ${current_price:.6f}\n\n"
        
        for entry, diff_percent in alerts:
            direction = "above" if current_price > entry else "below"
            message += f"📍 Entry: ${entry:.6f}\n"
            message += f"🎯 Distance: {diff_percent:.2f}% {direction}\n\n"
        
        message += f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"💡 Action: Consider your trading strategy!"
        
        return message
    
    def run_check_cycle(self):
        """Run one complete check cycle"""
        try:
            logger.info("Starting check cycle...")
            
            # Load sheet data
            df = self.load_sheet_data()
            if df is None or df.empty:
                logger.warning("No data loaded, skipping cycle")
                return
            
            # Identify symbol column
            symbol_col = None
            for col in ['Symbol', 'PAIR NAME', 'Pair', 'symbol', 'pair']:
                if col in df.columns:
                    symbol_col = col
                    break
            
            if symbol_col is None:
                logger.error("Could not find Symbol column")
                return
            
            # Get all symbols and prices
            symbols = df[symbol_col].tolist()
            price_data = self.get_batch_prices(symbols)
            
            alerts_sent = 0
            
            # Check each symbol
            for _, row in df.iterrows():
                symbol = row[symbol_col]
                current_price = price_data.get(symbol)
                
                if current_price is None:
                    continue
                
                # Parse entries
                entries = self.parse_entries(row)
                if not entries:
                    continue
                
                # Check proximity to entries
                proximity_alerts = self.check_entry_proximity(
                    current_price, entries, self.config.alert_threshold_percent
                )
                
                if proximity_alerts and self.should_send_alert(symbol):
                    # Send WhatsApp alert
                    message = self.format_alert_message(symbol, current_price, proximity_alerts)
                    
                    success = self.whatsapp.send_message(
                        self.config.target_phone_number, message
                    )
                    
                    if success:
                        alerts_sent += 1
                        logger.info(f"Alert sent for {symbol}")
                    else:
                        logger.error(f"Failed to send alert for {symbol}")
            
            # Clean expired cache
            self.price_cache.clear_expired()
            
            logger.info(f"Check cycle completed. Alerts sent: {alerts_sent}")
            
        except Exception as e:
            logger.error(f"Error in check cycle: {str(e)}")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        logger.info("Starting Crypto Alert System...")
        
        # Schedule the check every minute
        schedule.every(self.config.check_interval_minutes).minutes.do(self.run_check_cycle)
        
        # Send startup notification
        startup_message = f"🚀 Crypto Alert System Started!\n\n"
        startup_message += f"⚙️ Monitoring {self.config.check_interval_minutes} minute intervals\n"
        startup_message += f"🎯 Alert threshold: {self.config.alert_threshold_percent}%\n"
        startup_message += f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.whatsapp.send_message(self.config.target_phone_number, startup_message)
        
        # Run initial check
        self.run_check_cycle()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds for scheduled tasks
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    """Main function to run the alert system"""
    
    # Configuration - Use environment variables for security
    config = AlertConfig(
        sheet_url=os.getenv('GOOGLE_SHEET_URL', ''),
        whatsapp_api_token=os.getenv('WHATSAPP_API_TOKEN', ''),
        whatsapp_phone_number=os.getenv('WHATSAPP_PHONE_NUMBER', ''),
        target_phone_number=os.getenv('TARGET_PHONE_NUMBER', ''),
        alert_threshold_percent=float(os.getenv('ALERT_THRESHOLD', '1.0')),
        check_interval_minutes=int(os.getenv('CHECK_INTERVAL', '1'))
    )
    
    # Validate configuration
    if not all([config.sheet_url, config.whatsapp_api_token, 
                config.whatsapp_phone_number, config.target_phone_number]):
        logger.error("Missing required configuration. Please set environment variables:")
        logger.error("GOOGLE_SHEET_URL, WHATSAPP_API_TOKEN, WHATSAPP_PHONE_NUMBER, TARGET_PHONE_NUMBER")
        return
    
    # Create and start alert system
    alert_system = CryptoAlertSystem(config)
    alert_system.start_monitoring()

if __name__ == "__main__":
    main()

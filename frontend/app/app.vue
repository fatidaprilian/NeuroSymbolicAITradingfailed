<script setup>
import { ref, onMounted, onUnmounted, defineAsyncComponent, computed, watch } from 'vue'
import axios from 'axios'

// --- LAZY LOAD ApexChart ---
const ApexChart = defineAsyncComponent(() => import('vue3-apexcharts'))

// --- KONFIGURASI ---
const config = useRuntimeConfig()
const API_URL = config.public.apiBaseUrl
let timer = null

// --- STATE REAKTIF ---
const symbols = ref(['btc', 'eth', 'xrp'])
const selectedSymbol = ref(symbols.value[0])

const price = ref(0)
const prevPrice = ref(0)
const prediction = ref(0)
const changePercent = ref(0)
const balanceUSDT = ref(0)
const balanceCoin = ref(0)
const tradeHistory = ref([])
const lastUpdate = ref(new Date())
const errorMsg = ref('')
const loading = ref(true)
const isRefreshing = ref(false)

// Computed warna harga live
const priceColor = computed(() => {
  if (price.value > prevPrice.value) return 'text-green-500'
  if (price.value < prevPrice.value) return 'text-red-500'
  return 'text-gray-900 dark:text-white'
})

// --- OPSI GRAFIK (ApexCharts) ---
const chartSeries = ref([])
const chartOptions = ref({
  chart: {
    type: 'candlestick',
    height: 350,
    toolbar: { show: false },
    background: 'transparent',
    animations: { enabled: false }
  },
  theme: { mode: 'dark' },
  grid: {
    borderColor: '#334155',
    strokeDashArray: 3,
    xaxis: { lines: { show: false } }
  },
  xaxis: {
    type: 'datetime',
    axisBorder: { show: false },
    axisTicks: { show: false },
    labels: { style: { colors: '#64748b' } }
  },
  yaxis: {
    tooltip: { enabled: false },
    labels: {
      style: { colors: '#64748b' },
      formatter: v => '$' + v.toLocaleString()
    }
  },
  plotOptions: {
    candlestick: {
      colors: { upward: '#22c55e', downward: '#ef4444' }
    }
  }
})

// --- FUNGSI UTAMA: Fetch Data (Multi-Koin) ---
const fetchData = async (isBackground = false) => {
  if (!isBackground) loading.value = true
  else isRefreshing.value = true

  errorMsg.value = ''

  const symbol = selectedSymbol.value

  try {
    prevPrice.value = price.value

    const [resPredict, resHistory, resPortfolio, resTrades] = await Promise.all([
      axios.get(`${API_URL}/predict/live?symbol=${symbol}`),
      axios.get(`${API_URL}/market/history?symbol=${symbol}&hours=24`),
      axios.get(`${API_URL}/portfolio/testnet?symbol=${symbol}`),
      axios.get(`${API_URL}/trade/history?symbol=${symbol}&limit=10`)
    ])

    price.value = resPredict.data.current_price
    prediction.value = resPredict.data.prediction_next_hour
    changePercent.value = resPredict.data.predicted_change_percent

    chartSeries.value = [{ data: resHistory.data }]

    balanceUSDT.value = resPortfolio.data.usdt
    balanceCoin.value = resPortfolio.data[symbol] ?? 0

    tradeHistory.value = resTrades.data.map(trade => ({
      rawTime: new Date(trade.time),
      time: new Date(trade.time).toLocaleTimeString('id-ID', {
        hour: '2-digit',
        minute: '2-digit'
      }),
      side: trade.side,
      price: trade.price,
      qty: trade.qty,
      total: trade.price * trade.qty,
      status: 'FILLED'
    }))

    lastUpdate.value = new Date()
  } catch (err) {
    errorMsg.value = `Gagal mengambil data ${symbol.toUpperCase()}. Pastikan backend berjalan.`
    console.error(err)
  } finally {
    loading.value = false
    isRefreshing.value = false
  }
}

// --- LIFECYCLE ---
onMounted(() => {
  fetchData()
  timer = setInterval(() => fetchData(true), 60000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})

// --- WATCHER: ganti koin ---
watch(selectedSymbol, () => {
  fetchData(false)
})
</script>

<template>
  <div
    style="min-height: 100vh; background-color: rgb(249 250 251); color: rgb(17 24 39); font-family: system-ui, -apple-system, sans-serif; transition: all 0.3s;"
    class="dark:bg-gray-950 dark:text-gray-100"
  >
    <!-- HEADER -->
    <header
      style="position: sticky; top: 0; z-index: 50; border-bottom: 1px solid rgb(229 231 235); background-color: rgba(255 255 255 / 0.9); backdrop-filter: blur(20px); box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);"
      class="dark:border-gray-800 dark:bg-gray-900/90"
    >
      <div
        style="max-width: 80rem; margin: 0 auto; padding: 1.25rem 1rem; display: flex; justify-content: space-between; align-items: center;"
      >
        <div style="display: flex; align-items: center; gap: 1rem;">
          <div
            style="background: linear-gradient(to bottom right, rgb(99 102 241), rgb(79 70 229)); padding: 0.75rem; border-radius: 1rem; box-shadow: 0 10px 15px -3px rgb(99 102 241 / 0.25);"
          >
            <UIcon name="i-heroicons-cpu-chip" style="font-size: 1.5rem; color: white;" />
          </div>
          <div>
            <h1
              style="font-size: 1.25rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.025em; background: linear-gradient(to right, rgb(17 24 39), rgb(55 65 81)); background-clip: text; -webkit-background-clip: text; color: transparent;"
              class="dark:from-white dark:to-gray-300"
            >
              AutoTrader AI
            </h1>
            <div
              style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; font-weight: 600; color: rgb(107 114 128); margin-top: 0.25rem;"
            >
              <span
                style="position: relative; display: flex; height: 0.5rem; width: 0.5rem;"
              >
                <span
                  style="position: absolute; display: inline-flex; height: 100%; width: 100%; border-radius: 9999px; background-color: rgb(74 222 128); opacity: 0.75; animation: ping 1s cubic-bezier(0, 0, 0.2, 1) infinite;"
                ></span>
                <span
                  style="position: relative; display: inline-flex; border-radius: 9999px; height: 0.5rem; width: 0.5rem; background-color: rgb(34 197 94);"
                ></span>
              </span>
              <span>Live on Binance Testnet</span>
            </div>
          </div>
        </div>

        <!-- DESKTOP COIN SWITCHER -->
        <div
          style="display: none; align-items: center; gap: 0.25rem; padding: 0.25rem; background-color: rgb(243 244 246); border-radius: 0.5rem;"
          class="md:flex dark:bg-gray-800"
        >
          <button
            v-for="symbol in symbols"
            :key="symbol"
            @click="selectedSymbol = symbol"
            :style="{
              padding: '0.5rem 0.75rem',
              borderRadius: '0.375rem',
              fontSize: '0.875rem',
              fontWeight: '600',
              transition: 'all 0.2s',
              backgroundColor: selectedSymbol === symbol ? 'rgb(99 102 241)' : 'transparent',
              color: selectedSymbol === symbol ? 'white' : 'rgb(75 85 99)',
              cursor: 'pointer',
              border: 'none'
            }"
            :class="selectedSymbol !== symbol ? 'hover:bg-gray-200 dark:hover:bg-gray-700 dark:text-gray-300' : ''"
          >
            {{ symbol.toUpperCase() }}
          </button>
        </div>

        <!-- REFRESH -->
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          <div
            v-if="isRefreshing"
            style="font-size: 0.75rem; color: rgb(107 114 128); display: flex; align-items: center; gap: 0.5rem; background-color: rgb(243 244 246); padding: 0.5rem 0.75rem; border-radius: 0.5rem; font-weight: 500;"
            class="dark:bg-gray-800 dark:text-gray-400"
          >
            <UIcon
              name="i-heroicons-arrow-path"
              style="animation: spin 1s linear infinite; color: rgb(99 102 241);"
            />
            Syncing...
          </div>
          <UButton
            icon="i-heroicons-arrow-path"
            size="sm"
            color="primary"
            variant="soft"
            @click="fetchData(false)"
            :loading="loading"
            style="font-weight: 600;"
          >
            Refresh
          </UButton>
        </div>
      </div>
    </header>

    <!-- MAIN -->
    <div style="max-width: 80rem; margin: 0 auto; padding: 2.5rem 1rem;">
      <UAlert
        v-if="errorMsg"
        icon="i-heroicons-exclamation-triangle"
        color="red"
        variant="soft"
        :title="errorMsg"
        style="animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); margin-bottom: 2rem;"
      />

      <!-- MOBILE COIN SWITCHER -->
      <div style="display: flex; gap: 0.5rem; margin-bottom: 1.5rem;" class="md:hidden">
        <button
          v-for="symbol in symbols"
          :key="symbol"
          @click="selectedSymbol = symbol"
          :style="{
            flex: '1',
            padding: '0.75rem',
            borderRadius: '0.5rem',
            fontSize: '0.875rem',
            fontWeight: '700',
            transition: 'all 0.2s',
            backgroundColor:
              selectedSymbol === symbol ? 'rgb(99 102 241)' : 'rgb(243 244 246)',
            color: selectedSymbol === symbol ? 'white' : 'rgb(75 85 99)',
            cursor: 'pointer',
            border: 'none',
            boxShadow:
              selectedSymbol === symbol
                ? '0 4px 6px -1px rgb(99 102 241 / 0.3)'
                : 'none'
          }"
          :class="selectedSymbol === symbol ? '' : 'dark:bg-gray-800 dark:text-gray-300'"
        >
          {{ symbol.toUpperCase() }}
        </button>
      </div>

      <!-- TOP 3 CARDS -->
      <div
        style="display: grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap: 1.5rem; margin-bottom: 2rem;"
        class="md:grid-cols-3"
      >
        <!-- LIVE PRICE -->
        <UCard
          style="border: 1px solid rgb(229 231 235); background: linear-gradient(to bottom right, white, rgb(249 250 251 / 0.5), white); box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); transition: all 0.3s; cursor: pointer;"
          class="dark:border-gray-800 dark:from-gray-900 dark:via-gray-900/50 dark:to-gray-950 hover:shadow-xl hover:border-gray-300 dark:hover:border-gray-700 hover:-translate-y-1"
        >
          <div
            style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.25rem;"
          >
            <span
              style="font-size: 0.875rem; font-weight: 600; color: rgb(75 85 99); display: flex; align-items: center; gap: 0.5rem; background-color: rgb(243 244 246); padding: 0.375rem 0.75rem; border-radius: 0.5rem;"
              class="dark:text-gray-400 dark:bg-gray-800"
            >
              <UIcon
                name="i-heroicons-currency-dollar"
                style="color: rgb(107 114 128);"
              />
              {{ selectedSymbol.toUpperCase() }} Price
            </span>
            <UBadge
              color="gray"
              variant="solid"
              size="xs"
              style="font-weight: 700; letter-spacing: 0.1em; box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);"
            >
              LIVE
            </UBadge>
          </div>
          <div
            style="display: flex; align-items: baseline; gap: 0.75rem; margin-top: 0.5rem;"
          >
            <span
              style="font-size: 2.25rem; font-weight: 900; letter-spacing: -0.025em; transition: color 0.5s;"
              :class="priceColor"
            >
              <USkeleton
                v-if="loading"
                style="height: 3rem; width: 14rem; border-radius: 0.75rem;"
              />
              <span v-else>
                ${{ price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) }}
              </span>
            </span>
          </div>
          <div
            style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgb(229 231 235);"
            class="dark:border-gray-800"
          >
            <p
              style="font-size: 0.75rem; color: rgb(107 114 128); font-weight: 500;"
            >
              Last updated: {{ lastUpdate.toLocaleTimeString('id-ID') }}
            </p>
          </div>
        </UCard>

        <!-- AI FORECAST -->
        <UCard
          style="border: 2px solid rgb(99 102 241 / 0.3); position: relative; overflow: hidden; box-shadow: 0 10px 15px -3px rgb(99 102 241 / 0.2); transition: all 0.3s; cursor: pointer; background: linear-gradient(to bottom right, rgb(238 242 255), white);"
          class="dark:border-primary-400/20 dark:from-primary-950/30 dark:to-gray-900 hover:shadow-xl hover:shadow-primary-500/20 hover:-translate-y-1"
        >
          <div
            style="position: absolute; right: -2rem; top: -2rem; font-size: 8rem; opacity: 0.04; pointer-events: none; transform: rotate(12deg);"
            class="dark:opacity-[0.06]"
          >
            <UIcon name="i-heroicons-sparkles" />
          </div>

          <div
            style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.25rem; position: relative;"
          >
            <span
              style="font-size: 0.875rem; font-weight: 600; color: rgb(67 56 202); display: flex; align-items: center; gap: 0.5rem; background-color: rgb(224 231 255); padding: 0.375rem 0.75rem; border-radius: 0.5rem;"
              class="dark:text-primary-400 dark:bg-primary-900/50"
            >
              <UIcon name="i-heroicons-sparkles" />
              AI Forecast
            </span>
            <UBadge
              color="primary"
              variant="solid"
              size="xs"
              style="font-weight: 700; box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);"
            >
              1 HOUR
            </UBadge>
          </div>

          <div
            style="display: flex; flex-direction: column; gap: 0.75rem; position: relative;"
          >
            <div v-if="!loading">
              <div
                style="font-size: 2.25rem; font-weight: 900; letter-spacing: -0.025em; margin-bottom: 0.25rem;"
                :class="changePercent >= 0 ? 'text-green-600 dark:text-green-500' : 'text-red-600 dark:text-red-500'"
              >
                ${{ prediction.toLocaleString('en-US', { maximumFractionDigits: 0 }) }}
              </div>
              <div
                style="font-size: 1rem; font-weight: 700; margin-top: 0.75rem; display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0.75rem; border-radius: 0.5rem;"
                :class="changePercent >= 0 ? 'text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-950/30' : 'text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-950/30'"
              >
                <UIcon
                  :name="changePercent >= 0 ? 'i-heroicons-arrow-trending-up' : 'i-heroicons-arrow-trending-down'"
                  style="font-size: 1.25rem;"
                />
                {{ Math.abs(changePercent).toFixed(2) }}%
                {{ changePercent >= 0 ? 'UPSIDE' : 'DOWNSIDE' }}
              </div>
            </div>
            <USkeleton
              v-else
              style="height: 5rem; width: 100%; border-radius: 0.75rem;"
            />
          </div>
        </UCard>

        <!-- PORTFOLIO CARD -->
        <UCard
          style="border: 1px solid rgb(229 231 235); display: flex; flex-direction: column; justify-content: center; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); transition: all 0.3s; cursor: pointer; background: linear-gradient(to bottom right, white, rgb(249 250 251 / 0.3), white);"
          class="dark:border-gray-800 dark:from-gray-900 dark:via-gray-900/30 dark:to-gray-950 hover:shadow-xl hover:border-gray-300 dark:hover:border-gray-700 hover:-translate-y-1"
        >
          <div
            style="display: flex; justify-content: space-around; text-align: center; border-left: 2px solid rgb(229 231 235);"
            class="dark:border-gray-800"
          >
            <!-- USDT -->
            <div style="padding: 0 1.5rem; flex: 1;">
              <div
                style="font-size: 0.75rem; font-weight: 700; color: rgb(107 114 128); margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; display: flex; align-items: center; justify-content: center; gap: 0.375rem;"
              >
                <UIcon name="i-heroicons-banknotes" style="font-size: 1rem;" />
                USDT
              </div>
              <div
                style="font-weight: 900; font-size: 1.5rem; color: rgb(17 24 39);"
                class="dark:text-white"
              >
                <USkeleton
                  v-if="loading"
                  style="height: 2rem; width: 7rem; margin: 0 auto; border-radius: 0.5rem;"
                />
                <span v-else>
                  ${{ balanceUSDT.toLocaleString('en-US', { maximumFractionDigits: 0 }) }}
                </span>
              </div>
              <p
                style="font-size: 0.75rem; color: rgb(107 114 128); margin-top: 0.5rem; font-weight: 500;"
              >
                Available
              </p>
            </div>

            <!-- COIN -->
            <div
              style="padding: 0 1.5rem; flex: 1; border-left: 2px solid rgb(229 231 235);"
              class="dark:border-gray-800"
            >
              <div
                style="font-size: 0.75rem; font-weight: 700; color: rgb(107 114 128); margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; display: flex; align-items: center; justify-content: center; gap: 0.375rem;"
              >
                <UIcon :name="`i-cryptocurrency-${selectedSymbol}`" style="font-size: 1rem;" />
                {{ selectedSymbol.toUpperCase() }}
              </div>
              <div
                style="font-weight: 900; font-size: 1.5rem; color: rgb(17 24 39);"
                class="dark:text-white"
              >
                <USkeleton
                  v-if="loading"
                  style="height: 2rem; width: 7rem; margin: 0 auto; border-radius: 0.5rem;"
                />
                <span v-else>{{ balanceCoin.toFixed(4) }}</span>
              </div>
              <p
                style="font-size: 0.75rem; color: rgb(107 114 128); margin-top: 0.5rem; font-weight: 500;"
              >
                Holdings
              </p>
            </div>
          </div>
        </UCard>
      </div>

      <!-- BOTTOM GRID -->
      <div
        style="display: grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap: 2rem;"
        class="lg:grid-cols-12"
      >
        <!-- MARKET CHART -->
        <UCard
          style="padding: 0; overflow: hidden; border: 1px solid rgb(229 231 235); box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); transition: box-shadow 0.3s;"
          class="lg:col-span-8 dark:border-gray-800 hover:shadow-xl"
        >
          <template #header>
            <div
              style="display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.5rem; background-color: rgb(249 250 251 / 0.8); backdrop-filter: blur(8px);"
              class="dark:bg-gray-900/50"
            >
              <h3
                style="font-weight: 700; font-size: 1.125rem; color: rgb(31 41 55); display: flex; align-items: center; gap: 0.625rem;"
                class="dark:text-gray-100"
              >
                <div
                  style="background-color: rgb(99 102 241 / 0.1); padding: 0.5rem; border-radius: 0.5rem;"
                >
                  <UIcon
                    name="i-heroicons-chart-bar"
                    style="color: rgb(79 70 229); font-size: 1.125rem;"
                    class="dark:text-primary-400"
                  />
                </div>
                Market Chart ({{ selectedSymbol.toUpperCase() }})
              </h3>
              <div style="display: flex; gap: 0.5rem;">
                <UBadge
                  color="gray"
                  variant="soft"
                  size="sm"
                  style="font-weight: 600; padding: 0 0.75rem;"
                >
                  1H Candles
                </UBadge>
                <UBadge
                  color="primary"
                  variant="soft"
                  size="sm"
                  style="font-weight: 600; padding: 0 0.75rem;"
                >
                  24H History
                </UBadge>
              </div>
            </div>
          </template>

          <div
            style="padding: 1.5rem; min-height: 440px; background: linear-gradient(to bottom right, white, rgb(249 250 251 / 0.5));"
            class="dark:from-gray-900 dark:to-gray-950/50"
          >
            <ClientOnly>
              <ApexChart
                v-if="!loading && chartSeries.length > 0"
                height="400"
                type="candlestick"
                :options="chartOptions"
                :series="chartSeries"
              />
              <div
                v-else
                style="height: 400px; display: flex; align-items: center; justify-content: center; color: rgb(156 163 175); flex-direction: column; gap: 1rem;"
              >
                <UIcon
                  name="i-heroicons-arrow-path"
                  style="animation: spin 1s linear infinite; font-size: 2.25rem; color: rgb(99 102 241 / 0.5);"
                />
                <span style="font-weight: 600; font-size: 1.125rem;">
                  Loading market data...
                </span>
              </div>
            </ClientOnly>
          </div>
        </UCard>

        <!-- TRADE HISTORY -->
        <UCard
          style="padding: 0; border: 1px solid rgb(229 231 235); display: flex; flex-direction: column; height: 100%; box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); transition: box-shadow 0.3s;"
          class="lg:col-span-4 dark:border-gray-800 hover:shadow-xl"
        >
          <template #header>
            <div
              style="display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.5rem; background-color: rgb(249 250 251 / 0.8); backdrop-filter: blur(8px);"
              class="dark:bg-gray-900/50"
            >
              <h3
                style="font-weight: 700; font-size: 1.125rem; color: rgb(31 41 55); display: flex; align-items: center; gap: 0.625rem;"
                class="dark:text-gray-100"
              >
                <div
                  style="background-color: rgb(99 102 241 / 0.1); padding: 0.5rem; border-radius: 0.5rem;"
                >
                  <UIcon
                    name="i-heroicons-clipboard-document-list"
                    style="color: rgb(79 70 229); font-size: 1.125rem;"
                    class="dark:text-primary-400"
                  />
                </div>
                Trade History ({{ selectedSymbol.toUpperCase() }})
              </h3>
              <UBadge
                :color="tradeHistory.length > 0 ? 'primary' : 'gray'"
                variant="subtle"
                size="sm"
                style="font-weight: 700; padding: 0 0.75rem;"
              >
                {{ tradeHistory.length }} Trades
              </UBadge>
            </div>
          </template>

          <!-- EMPTY STATE -->
          <div
            v-if="!loading && tradeHistory.length === 0"
            style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2.5rem; text-align: center; min-height: 340px; background: linear-gradient(to bottom right, white, rgb(249 250 251 / 0.5));"
            class="dark:from-gray-900 dark:to-gray-950/50"
          >
            <div
              style="background: linear-gradient(to bottom right, rgb(243 244 246), rgb(229 231 235)); padding: 2rem; border-radius: 1rem; margin-bottom: 1.25rem; box-shadow: inset 0 2px 4px 0 rgb(0 0 0 / 0.05);"
              class="dark:from-gray-800 dark:to-gray-800/50"
            >
              <UIcon
                name="i-heroicons-inbox"
                style="font-size: 3.75rem; color: rgb(209 213 219);"
                class="dark:text-gray-600"
              />
            </div>
            <h4
              style="font-weight: 700; font-size: 1.25rem; color: rgb(17 24 39); margin-bottom: 0.5rem;"
              class="dark:text-white"
            >
              No Trades Yet
            </h4>
            <p
              style="font-size: 0.875rem; color: rgb(107 114 128); max-width: 220px; line-height: 1.625;"
            >
              Bot is waiting for optimal market signals to execute trades.
            </p>
          </div>

          <!-- TABLE -->
          <div
            v-else
            style="overflow-y: auto; max-height: 456px; background-color: white;"
            class="custom-scrollbar dark:bg-gray-950"
          >
            <table style="width: 100%; font-size: 0.875rem;">
              <thead
                style="font-size: 0.75rem; font-weight: 700; color: rgb(75 85 99); text-transform: uppercase; background-color: rgb(243 244 246 / 0.8); backdrop-filter: blur(8px); position: sticky; top: 0; border-bottom: 2px solid rgb(229 231 235);"
                class="dark:text-gray-400 dark:bg-gray-900/80 dark:border-gray-800"
              >
                <tr>
                  <th style="padding: 1rem 1.5rem; text-align: left;">Action</th>
                  <th style="padding: 1rem 1.5rem; text-align: right;">Price / Total</th>
                  <th style="padding: 1rem 1.5rem; text-align: right;">Time</th>
                </tr>
              </thead>
              <tbody
                style="border-top: 1px solid rgb(243 244 246);"
                class="dark:divide-gray-800/50"
              >
                <tr
                  v-for="(trade, i) in tradeHistory"
                  :key="i"
                  style="border-bottom: 1px solid rgb(243 244 246); transition: background-color 0.2s;"
                  class="dark:border-gray-800/50 hover:bg-gray-50 dark:hover:bg-gray-800/40"
                >
                  <td style="padding: 1rem 1.5rem;">
                    <div
                      style="display: flex; flex-direction: column; align-items: flex-start; gap: 0.5rem;"
                    >
                      <UBadge
                        :color="trade.side === 'BUY' ? 'green' : 'red'"
                        variant="subtle"
                        size="sm"
                        style="font-weight: 700; padding: 0 0.75rem; box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);"
                      >
                        {{ trade.side }}
                      </UBadge>
                      <span
                        style="font-size: 0.75rem; font-weight: 600; color: rgb(75 85 99);"
                        class="dark:text-gray-400"
                      >
                        {{ trade.qty.toFixed(4) }} {{ selectedSymbol.toUpperCase() }}
                      </span>
                    </div>
                  </td>
                  <td style="padding: 1rem 1.5rem; text-align: right;">
                    <div
                      style="font-weight: 700; font-size: 1rem; color: rgb(17 24 39);"
                      class="dark:text-gray-100"
                    >
                      ${{ trade.price.toLocaleString('en-US', { maximumFractionDigits: 0 }) }}
                    </div>
                    <div
                      style="font-size: 0.75rem; color: rgb(107 114 128); margin-top: 0.25rem; font-weight: 500;"
                    >
                      ≈
                      ${{ trade.total.toLocaleString('en-US', { maximumFractionDigits: 2 }) }}
                    </div>
                  </td>
                  <td
                    style="padding: 1rem 1.5rem; text-align: right; font-size: 0.75rem; font-weight: 600; color: rgb(107 114 128);"
                  >
                    {{ trade.time }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </UCard>
      </div>
    </div>
  </div>
</template>

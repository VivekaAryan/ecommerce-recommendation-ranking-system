import { useCallback, useEffect, useState } from "react";
import { api, type ProductCard } from "./api";
import type { RecommendResponse, SystemStatus, UserInfo } from "./types";
import "./RecommendShop.css";

interface RecommendShopProps {
  status: SystemStatus | null;
  users: UserInfo[];
  usersError: string | null;
  selectedUser: string;
  onUserChange: (userId: string) => void;
  onRefreshUsers: () => void;
}

function formatPrice(price: number | null) {
  if (price == null) return "—";
  return `$${price.toFixed(2)}`;
}

function ProductImage({ product, className = "" }: { product: ProductCard; className?: string }) {
  const [failed, setFailed] = useState(false);
  const fallback = `https://placehold.co/600x600/f5f5f5/666666?text=${encodeURIComponent(product.category)}`;
  return (
    <img
      className={className}
      src={failed ? fallback : product.image_url}
      alt={product.title}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

export default function RecommendShop({
  status,
  users,
  usersError,
  selectedUser,
  onUserChange,
  onRefreshUsers,
}: RecommendShopProps) {
  const [catalog, setCatalog] = useState<ProductCard[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [activeCategory, setActiveCategory] = useState<string>("All");
  const [selectedProduct, setSelectedProduct] = useState<ProductCard | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendResponse | null>(null);
  const [loadingCatalog, setLoadingCatalog] = useState(false);
  const [loadingRec, setLoadingRec] = useState(false);
  const [shopError, setShopError] = useState<string | null>(null);
  const [showLatency, setShowLatency] = useState(false);

  const loadCatalog = useCallback(async (category?: string) => {
    setLoadingCatalog(true);
    setShopError(null);
    try {
      const cat = category && category !== "All" ? category : undefined;
      const data = await api.catalog(48, 0, cat);
      setCatalog(data.products);
      setCategories(data.categories);
    } catch (e) {
      setShopError(e instanceof Error ? e.message : "Failed to load catalog");
      setCatalog([]);
    } finally {
      setLoadingCatalog(false);
    }
  }, []);

  const closeModal = useCallback(() => {
    setSelectedProduct(null);
    setRecommendation(null);
    setShowLatency(false);
    setLoadingRec(false);
  }, []);

  useEffect(() => {
    loadCatalog(activeCategory);
  }, [activeCategory, loadCatalog]);

  useEffect(() => {
    closeModal();
  }, [selectedUser, closeModal]);

  useEffect(() => {
    if (!selectedProduct) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeModal();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedProduct, closeModal]);

  const handleProductClick = async (product: ProductCard) => {
    if (!selectedUser) return;
    setSelectedProduct(product);
    setRecommendation(null);
    setShowLatency(false);
    setLoadingRec(true);
    setShopError(null);
    try {
      const result = await api.recommend(selectedUser, product.item_id);
      setRecommendation(result);
    } catch (e) {
      setShopError(e instanceof Error ? e.message : "Recommendation failed");
    } finally {
      setLoadingRec(false);
    }
  };

  const budgetMs = { retrieval: 20, prerank: 10, ranking: 50, reranking: 30, total: 100 };

  return (
    <div className="shop">
      <header className="shop-header">
        <div className="shop-brand">
          <span className="shop-logo">⚡</span>
          <div>
            <h1>VoltMart</h1>
            <p>Appliances, Books, Electronics &amp; more</p>
          </div>
        </div>
        <div className="shop-search">
          <input type="search" placeholder="Search products across categories..." disabled />
        </div>
        <div className="shop-user-bar">
          <label className="shop-user-label">Shopping as</label>
          <select
            className="shop-user-select"
            value={selectedUser}
            onChange={(e) => onUserChange(e.target.value)}
          >
            <option value="">Select account...</option>
            {users.map((u) => (
              <option key={u.user_id} value={u.user_id}>
                {u.user_id} ({u.interaction_count} orders)
              </option>
            ))}
          </select>
          <button type="button" className="shop-btn-ghost" onClick={onRefreshUsers}>
            Refresh
          </button>
        </div>
      </header>

      {(usersError || shopError) && (
        <div className="shop-alert">
          {usersError || shopError}
        </div>
      )}

      {!selectedUser && (
        <div className="shop-empty">
          <h2>Welcome to VoltMart</h2>
          <p>Select an account above to browse and get personalized recommendations.</p>
          {!status?.ready && (
            <p className="shop-hint">Run the Download pipeline step first to prepare the catalog.</p>
          )}
        </div>
      )}

      {selectedUser && (
        <>
          <section className="shop-hero">
            <div className="shop-hero-text">
              <span className="shop-hero-tag">7 Amazon categories</span>
              <h2>Shop real products, get personalized picks</h2>
              <p>Click any product to open recommendations powered by our retrieval → rank → re-rank pipeline.</p>
            </div>
          </section>

          <section className="shop-section">
            <div className="shop-section-head">
              <h3>Browse catalog</h3>
              <div className="shop-categories">
                <button
                  type="button"
                  className={activeCategory === "All" ? "active" : ""}
                  onClick={() => setActiveCategory("All")}
                >
                  All
                </button>
                {categories.map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    className={activeCategory === cat ? "active" : ""}
                    onClick={() => setActiveCategory(cat)}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {loadingCatalog ? (
              <p className="shop-loading">Loading products...</p>
            ) : (
              <div className="shop-grid">
                {catalog.map((p) => (
                  <button
                    key={p.item_id}
                    type="button"
                    className={`shop-product ${selectedProduct?.item_id === p.item_id ? "selected" : ""}`}
                    onClick={() => handleProductClick(p)}
                  >
                    <div className="shop-product-img-wrap">
                      <ProductImage product={p} className="shop-product-img" />
                      {loadingRec && selectedProduct?.item_id === p.item_id && (
                        <div className="shop-product-overlay">Finding picks...</div>
                      )}
                    </div>
                    <div className="shop-product-body">
                      <span className="shop-product-cat">{p.category}</span>
                      <h4 title={p.title}>{p.title}</h4>
                      <div className="shop-product-footer">
                        <span className="shop-product-price">{formatPrice(p.price)}</span>
                        <span className="shop-product-cta">View picks →</span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </section>
        </>
      )}

      {selectedProduct && (
        <div className="shop-modal-backdrop" onClick={closeModal} role="presentation">
          <div
            className="shop-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="shop-modal-title"
          >
            <header className="shop-modal-header">
              <div className="shop-modal-product">
                <ProductImage product={selectedProduct} className="shop-modal-thumb" />
                <div>
                  <span className="shop-product-cat">{selectedProduct.category}</span>
                  <h2 id="shop-modal-title">{selectedProduct.title}</h2>
                  <p className="shop-modal-price">{formatPrice(selectedProduct.price)}</p>
                </div>
              </div>
              <button type="button" className="shop-modal-close" onClick={closeModal} aria-label="Close">
                ×
              </button>
            </header>

            <div className="shop-modal-body">
              <div className="shop-rec-head">
                <h3>Recommended for you</h3>
                {loadingRec && <span className="shop-rec-loading">Running retrieval → rank → re-rank...</span>}
              </div>

              {!loadingRec && recommendation && recommendation.slate.length > 0 && (
                <div className="shop-rec-grid">
                  {recommendation.slate.map((item) => (
                    <article key={item.item_id} className="shop-rec-card">
                      <div className="shop-rec-rank">#{item.position + 1}</div>
                      <ProductImage
                        product={{
                          item_id: item.item_id,
                          title: item.title,
                          category: item.category,
                          price: item.price,
                          image_url: item.image_url,
                        }}
                        className="shop-rec-img"
                      />
                      <div className="shop-rec-body">
                        <span className="shop-product-cat">{item.category}</span>
                        <h4 title={item.title}>{item.title}</h4>
                        <div className="shop-rec-meta">
                          <span className="shop-product-price">{formatPrice(item.price)}</span>
                          <span className="shop-rec-score">match {item.score.toFixed(2)}</span>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              )}

              {!loadingRec && recommendation && recommendation.slate.length === 0 && (
                <p className="shop-empty-inline">No recommendations returned. Train models via the Pipeline tab.</p>
              )}

              {recommendation && (
                <div className="shop-dev-panel">
                  <button
                    type="button"
                    className="shop-dev-toggle"
                    onClick={() => setShowLatency((v) => !v)}
                  >
                    {showLatency ? "Hide" : "Show"} pipeline latency
                  </button>
                  {showLatency && (
                    <div className="shop-latency">
                      {(["retrieval", "prerank", "ranking", "reranking", "total"] as const).map((stage) => {
                        const ms = recommendation.latency[`${stage}_ms` as keyof typeof recommendation.latency] as number;
                        const budget = budgetMs[stage];
                        const ok = recommendation.latency.within_budget[stage];
                        return (
                          <div key={stage} className="shop-latency-row">
                            <span>{stage}</span>
                            <div className="shop-latency-bar">
                              <div
                                className={ok ? "ok" : "over"}
                                style={{ width: `${Math.min((ms / budget) * 100, 100)}%` }}
                              />
                            </div>
                            <span>{ms.toFixed(1)}ms</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
